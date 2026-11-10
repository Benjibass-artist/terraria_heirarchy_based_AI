using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;

namespace TerrariaAgent;

public sealed class AgentAction
{
    [JsonPropertyName("left")]
    public bool Left { get; set; }

    [JsonPropertyName("right")]
    public bool Right { get; set; }

    [JsonPropertyName("jump")]
    public bool Jump { get; set; }

    [JsonPropertyName("down")]
    public bool Down { get; set; }

    [JsonPropertyName("attack")]
    public bool Attack { get; set; }

    [JsonPropertyName("aim_x")]
    public float AimX { get; set; } = 1f;

    [JsonPropertyName("aim_y")]
    public float AimY { get; set; }

    [JsonPropertyName("auto_aim")]
    public bool AutoAim { get; set; }

    [JsonPropertyName("target_slot")]
    public int TargetSlot { get; set; }

    [JsonPropertyName("hotbar_slot")]
    public int HotbarSlot { get; set; }

    [JsonPropertyName("hotbar_control")]
    public bool HotbarControl { get; set; }
}

public sealed class CombatObservation
{
    [JsonPropertyName("tick")]
    public ulong Tick { get; set; }

    [JsonPropertyName("grid_flags")]
    public int[] GridFlags { get; set; } = [];

    [JsonPropertyName("enemy_type_ids")]
    public int[] EnemyTypeIds { get; set; } = [];

    [JsonPropertyName("player_features")]
    public float[] PlayerFeatures { get; set; } = [];

    [JsonPropertyName("weapon_features")]
    public float[] WeaponFeatures { get; set; } = [];

    [JsonPropertyName("weapon_type_id")]
    public int WeaponTypeId { get; set; }

    [JsonPropertyName("weapon_name")]
    public string WeaponName { get; set; } = "unknown";

    [JsonPropertyName("hotbar")]
    public List<HotbarSlotObservation> Hotbar { get; set; } = new();

    [JsonPropertyName("attack_attempted")]
    public bool AttackAttempted { get; set; }

    [JsonPropertyName("attack_hit")]
    public bool AttackHit { get; set; }

    [JsonPropertyName("attack_not_ready")]
    public bool AttackNotReady { get; set; }

    [JsonPropertyName("damage_taken")]
    public float DamageTaken { get; set; }

    [JsonPropertyName("enemies_killed")]
    public int EnemiesKilled { get; set; }

    [JsonPropertyName("was_alive")]
    public bool WasAlive { get; set; }

    [JsonPropertyName("is_alive")]
    public bool IsAlive { get; set; }

    [JsonPropertyName("enemies")]
    public List<EnemyObservation> Enemies { get; set; } = new();
}

public sealed class HotbarSlotObservation
{
    [JsonPropertyName("slot")]
    public int Slot { get; set; }

    [JsonPropertyName("item_type_id")]
    public int ItemTypeId { get; set; }

    [JsonPropertyName("valid_weapon")]
    public float ValidWeapon { get; set; }

    [JsonPropertyName("damage")]
    public float Damage { get; set; }

    [JsonPropertyName("use_time")]
    public float UseTime { get; set; }

    [JsonPropertyName("knockback")]
    public float Knockback { get; set; }

    [JsonPropertyName("shoot")]
    public float Shoot { get; set; }

    [JsonPropertyName("shoot_speed")]
    public float ShootSpeed { get; set; }

    [JsonPropertyName("melee")]
    public float Melee { get; set; }

    [JsonPropertyName("no_melee")]
    public float NoMelee { get; set; }

    [JsonPropertyName("range")]
    public float Range { get; set; }

    [JsonPropertyName("area")]
    public float Area { get; set; }

    [JsonPropertyName("selected")]
    public float Selected { get; set; }
}

public sealed class EnemyObservation
{
    [JsonPropertyName("type_id")]
    public int TypeId { get; set; }

    [JsonPropertyName("relative_x")]
    public float RelativeX { get; set; }

    [JsonPropertyName("relative_y")]
    public float RelativeY { get; set; }

    [JsonPropertyName("velocity_x")]
    public float VelocityX { get; set; }

    [JsonPropertyName("velocity_y")]
    public float VelocityY { get; set; }

    [JsonPropertyName("health_fraction")]
    public float HealthFraction { get; set; }

    [JsonPropertyName("current_life")]
    public int CurrentLife { get; set; }

    [JsonPropertyName("max_life")]
    public int MaxLife { get; set; }

    [JsonPropertyName("damage")]
    public int Damage { get; set; }

    [JsonPropertyName("defense")]
    public int Defense { get; set; }

    [JsonPropertyName("line_of_sight")]
    public float LineOfSight { get; set; }

    [JsonPropertyName("threat")]
    public float Threat { get; set; }

    [JsonPropertyName("distance")]
    public float Distance { get; set; }
}

internal sealed class AgentTcpClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);
    private readonly string host;
    private readonly int port;
    private readonly object outboundLock = new();
    private readonly ConcurrentQueue<AgentAction> inbound = new();
    private readonly AutoResetEvent workAvailable = new(false);
    private readonly CancellationTokenSource cancellation = new();
    private readonly Thread worker;
    private string? latestOutbound;

    public AgentTcpClient(string host, int port)
    {
        this.host = host;
        this.port = port;
        worker = new Thread(Run) { IsBackground = true, Name = "TerrariaAgentTcp" };
        worker.Start();
    }

    public void Send(CombatObservation observation)
    {
        string serialized = JsonSerializer.Serialize(observation, JsonOptions);
        lock (outboundLock)
        {
            latestOutbound = serialized;
        }
        workAvailable.Set();
    }

    public bool TryGetAction(out AgentAction action)
    {
        AgentAction? latest = null;
        while (inbound.TryDequeue(out AgentAction? next))
        {
            latest = next;
        }

        action = latest!;
        return latest is not null;
    }

    private void ClearInboundActions()
    {
        while (inbound.TryDequeue(out _))
        {
        }
    }

    public void Dispose()
    {
        cancellation.Cancel();
        workAvailable.Set();
        worker.Join(TimeSpan.FromSeconds(1));
        workAvailable.Dispose();
        cancellation.Dispose();
    }

    private void Run()
    {
        while (!cancellation.IsCancellationRequested)
        {
            try
            {
                using TcpClient tcpClient = new();
                tcpClient.Connect(host, port);
                using NetworkStream stream = tcpClient.GetStream();
                using StreamReader reader = new(stream, Encoding.UTF8, leaveOpen: true);
                using StreamWriter writer = new(stream, new UTF8Encoding(false), leaveOpen: true)
                {
                    AutoFlush = true,
                    NewLine = "\n",
                };

                while (!cancellation.IsCancellationRequested && tcpClient.Connected)
                {
                    string? message;
                    lock (outboundLock)
                    {
                        message = latestOutbound;
                        latestOutbound = null;
                    }

                    if (message is null)
                    {
                        workAvailable.WaitOne(50);
                        continue;
                    }

                    writer.WriteLine(message);
                    string? response = reader.ReadLine();
                    if (response is null)
                    {
                        break;
                    }

                    AgentAction? action = JsonSerializer.Deserialize<AgentAction>(response, JsonOptions);
                    if (action is not null)
                    {
                        inbound.Enqueue(action);
                    }
                }
            }
            catch (SocketException)
            {
                ClearInboundActions();
                workAvailable.WaitOne(500);
            }
            catch (IOException)
            {
                ClearInboundActions();
                workAvailable.WaitOne(500);
            }
        }
    }
}
