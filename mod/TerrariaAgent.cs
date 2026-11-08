using Terraria;
using Terraria.ModLoader;

namespace TerrariaAgent;

public sealed class TerrariaAgent : Mod
{
    internal static AgentTcpClient? Client { get; private set; }

    public override void Load()
    {
        if (Main.dedServ)
        {
            return;
        }

        Client = new AgentTcpClient("127.0.0.1", 8765);
    }

    public override void Unload()
    {
        Client?.Dispose();
        Client = null;
    }
}
