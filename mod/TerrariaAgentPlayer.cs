using System;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;
using Terraria.ModLoader;

namespace TerrariaAgent;

public sealed class TerrariaAgentPlayer : ModPlayer
{
    private static readonly Vector2[] AimDirections =
    [
        new Vector2(1f, 0f),
        new Vector2(1f, 1f),
        new Vector2(0f, 1f),
        new Vector2(-1f, 1f),
        new Vector2(-1f, 0f),
        new Vector2(-1f, -1f),
        new Vector2(0f, -1f),
        new Vector2(1f, -1f),
    ];
    private const int GridWidth = 80;
    private const int GridHeight = 45;
    private AgentAction currentAction = new();
    private bool wasAlive = true;
    private int previousLife;
    private int previousItemAnimation;
    private bool attackInProgress;
    private bool attackHitThisSwing;
    private bool attackAttemptedThisTick;
    private bool attackHitThisTick;
    private int enemiesKilledThisTick;

    public override void OnEnterWorld()
    {
        wasAlive = !Player.dead;
        previousLife = Player.statLife;
        previousItemAnimation = Player.itemAnimation;
    }

    public override void PreUpdateMovement()
    {
        if (Main.gameMenu || Main.LocalPlayer != Player)
        {
            return;
        }

        if (TerrariaAgent.Client?.TryGetAction(out AgentAction? action) == true)
        {
            currentAction = action;
        }

        Player.controlLeft = currentAction.Left;
        Player.controlRight = currentAction.Right;
        Player.controlJump = currentAction.Jump;
        Player.controlDown = currentAction.Down;
        Player.controlUseItem = currentAction.Attack;
        Vector2 aimDirection = AimDirections[Math.Clamp(currentAction.AimDirection, 0, 7)];
        if (aimDirection.X != 0f)
        {
            Player.direction = aimDirection.X > 0f ? 1 : -1;
        }
        Player.itemRotation = aimDirection.ToRotation();
    }

    public override void PostUpdate()
    {
        if (Main.gameMenu || Main.LocalPlayer != Player)
        {
            return;
        }

        if (Player.itemAnimation > previousItemAnimation)
        {
            attackInProgress = true;
            attackHitThisSwing = false;
        }

        bool attackCompleted = attackInProgress && previousItemAnimation > 0 && Player.itemAnimation == 0;
        attackAttemptedThisTick = attackCompleted;
        attackHitThisTick = attackCompleted && attackHitThisSwing;
        if (attackCompleted)
        {
            attackInProgress = false;
        }

        CombatObservation observation = BuildObservation();
        TerrariaAgent.Client?.Send(observation);
        wasAlive = !Player.dead;
        previousLife = Player.statLife;
        previousItemAnimation = Player.itemAnimation;
        attackAttemptedThisTick = false;
        attackHitThisTick = false;
        enemiesKilledThisTick = 0;
    }

    private CombatObservation BuildObservation()
    {
        int[] gridFlags = new int[GridWidth * GridHeight];
        int[] enemyTypeIds = new int[GridWidth * GridHeight];
        Point playerTile = Player.Center.ToTileCoordinates();
        int left = playerTile.X - GridWidth / 2;
        int top = playerTile.Y - GridHeight / 2;

        for (int localY = 0; localY < GridHeight; localY++)
        {
            for (int localX = 0; localX < GridWidth; localX++)
            {
                int worldX = left + localX;
                int worldY = top + localY;
                int index = localY * GridWidth + localX;
                if (!WorldGen.InWorld(worldX, worldY, 1))
                {
                    continue;
                }

                Tile tile = Main.tile[worldX, worldY];
                int flags = 0;
                if (tile.HasTile)
                {
                    flags |= 1;
                    if (TileID.Sets.Platforms[tile.TileType])
                    {
                        flags |= 2;
                    }
                }
                if (tile.LiquidAmount > 0)
                {
                    flags |= 4;
                }
                gridFlags[index] = flags;
            }
        }

        for (int npcIndex = 0; npcIndex < Main.maxNPCs; npcIndex++)
        {
            NPC npc = Main.npc[npcIndex];
            if (!npc.active || !npc.CanBeChasedBy())
            {
                continue;
            }

            Point npcTile = npc.Center.ToTileCoordinates();
            int localX = npcTile.X - left;
            int localY = npcTile.Y - top;
            if (localX < 0 || localX >= GridWidth || localY < 0 || localY >= GridHeight)
            {
                continue;
            }

            int index = localY * GridWidth + localX;
            enemyTypeIds[index] = Math.Min(npc.type + 1, 511);
            gridFlags[index] |= 8;
        }

        return new CombatObservation
        {
            Tick = (ulong)Main.GameUpdateCount,
            GridFlags = gridFlags,
            EnemyTypeIds = enemyTypeIds,
            PlayerFeatures = BuildPlayerFeatures(playerTile),
            WeaponFeatures = BuildWeaponFeatures(),
            AttackAttempted = attackAttemptedThisTick,
            AttackHit = attackHitThisTick,
            DamageTaken = Math.Max(previousLife - Player.statLife, 0),
            EnemiesKilled = enemiesKilledThisTick,
            WasAlive = wasAlive,
            IsAlive = !Player.dead,
        };
    }

    public override void OnHitNPC(NPC target, NPC.HitInfo hit, int damageDone)
    {
        attackHitThisSwing = true;
        if (target.life <= 0)
        {
            enemiesKilledThisTick++;
        }
    }

    private float[] BuildPlayerFeatures(Point playerTile)
    {
        return
        [
            Player.statLife / (float)Math.Max(Player.statLifeMax2, 1),
            Player.statMana / (float)Math.Max(Player.statManaMax2, 1),
            Player.velocity.X / 16f,
            Player.velocity.Y / 16f,
            Player.direction,
            Player.dead ? 1f : 0f,
            playerTile.X,
            playerTile.Y,
            wasAlive ? 1f : 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f,
        ];
    }

    private float[] BuildWeaponFeatures()
    {
        Item item = Player.HeldItem;
        return
        [
            item.damage,
            item.useTime,
            item.useAnimation,
            item.knockBack,
            item.shoot,
            item.shootSpeed,
            item.DamageType.CountsAsClass(DamageClass.Melee) ? 1f : 0f,
            item.noMelee ? 0f : 1f,
            item.axe,
            item.pick,
            item.hammer,
            Player.selectedItem,
            0f, 0f, 0f, 0f,
        ];
    }
}
