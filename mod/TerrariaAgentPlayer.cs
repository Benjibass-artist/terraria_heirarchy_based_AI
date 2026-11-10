using System;
using System.Collections.Generic;
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
    private const int MaxActionAge = 30;
    private AgentAction currentAction = new();
    private int actionAge = MaxActionAge;
    private bool wasAlive = true;
    private int previousLife;
    private int previousItemAnimation;
    private bool attackInProgress;
    private bool attackHitThisSwing;
    private bool attackAttemptedThisTick;
    private bool attackHitThisTick;
    private bool attackNotReadyThisTick;
    private bool previousAttackRequest;
    private int enemiesKilledThisTick;

    public override void OnEnterWorld()
    {
        wasAlive = !Player.dead;
        previousLife = Player.statLife;
        previousItemAnimation = Player.itemAnimation;
    }

    public override void SetControls()
    {
        if (Main.gameMenu || Main.LocalPlayer != Player)
        {
            return;
        }

        if (TerrariaAgent.Client?.TryGetAction(out AgentAction? action) == true)
        {
            currentAction = action;
            actionAge = 0;
        }
        else
        {
            actionAge++;
            if (actionAge > MaxActionAge)
            {
                currentAction = new AgentAction();
            }
        }

        bool actionIsFresh = actionAge <= MaxActionAge;
        if (
            actionIsFresh
            && currentAction.HotbarControl
            && currentAction.HotbarSlot >= 0
            && currentAction.HotbarSlot < 10
            && IsCombatItem(Player.inventory[currentAction.HotbarSlot])
        )
        {
            Player.selectedItem = currentAction.HotbarSlot;
        }
        bool hasCombatItem = IsCombatItem(Player.HeldItem);
        bool attackRequested = actionIsFresh && currentAction.Attack && hasCombatItem;
        bool weaponReady = hasCombatItem
            && Player.itemTime <= 0
            && Player.itemAnimation <= 0
            && !Player.dead;
        if (attackRequested && !previousAttackRequest && !weaponReady)
        {
            attackNotReadyThisTick = true;
        }
        previousAttackRequest = attackRequested;
        bool shouldAttack = attackRequested && weaponReady;
        Player.controlLeft = actionIsFresh && currentAction.Left;
        Player.controlRight = actionIsFresh && currentAction.Right;
        Player.controlJump = actionIsFresh && currentAction.Jump;
        Player.controlDown = actionIsFresh && currentAction.Down;
        Player.controlUseItem = shouldAttack;
        Player.releaseUseItem = true;
        if (shouldAttack)
        {
            Player.controlUseTile = false;
        }
        Vector2 aimDirection = new(currentAction.AimX, currentAction.AimY);
        if (aimDirection.LengthSquared() < 0.0001f)
        {
            aimDirection = Vector2.UnitX * Player.direction;
        }
        else
        {
            aimDirection.Normalize();
        }
        if (currentAction.AutoAim && TryGetTargetDirection(currentAction.TargetSlot, out Vector2 targetDirection))
        {
            aimDirection = targetDirection;
        }
        SetVirtualCursor(aimDirection);
        if (aimDirection.X != 0f)
        {
            Player.direction = aimDirection.X > 0f ? 1 : -1;
        }
        Player.itemRotation = aimDirection.ToRotation();
    }

    public override void PreUpdateMovement()
    {
        if (Main.gameMenu || Main.LocalPlayer != Player || Main.instance.IsActive)
        {
            return;
        }

        if (actionAge > MaxActionAge || Player.dead)
        {
            return;
        }

        int horizontalDirection = (currentAction.Right ? 1 : 0) - (currentAction.Left ? 1 : 0);
        if (horizontalDirection != 0)
        {
            Player.velocity.X = horizontalDirection * Math.Max(Player.maxRunSpeed, 3f);
        }

        if (currentAction.Jump && (Player.velocity.Y == 0f || Player.collideY))
        {
            Player.velocity.Y = -Player.jumpSpeed;
        }
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
        attackNotReadyThisTick = false;
        enemiesKilledThisTick = 0;
    }

    private static bool IsCombatItem(Item item)
    {
        return item.damage > 0 && item.pick == 0 && item.axe == 0 && item.hammer == 0;
    }

    private CombatObservation BuildObservation()
    {
        int[] gridFlags = new int[GridWidth * GridHeight];
        int[] enemyTypeIds = new int[GridWidth * GridHeight];
        List<EnemyObservation> enemies = new();
        Vector2 closestEnemyOffset = Vector2.Zero;
        float closestEnemyDistanceSquared = float.MaxValue;
        bool hasClosestEnemy = false;
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
            Vector2 enemyOffset = npc.Center - Player.Center;
            float enemyDistanceSquared = enemyOffset.LengthSquared();
            enemies.Add(new EnemyObservation
            {
                TypeId = npc.type,
                RelativeX = enemyOffset.X / 16f,
                RelativeY = enemyOffset.Y / 16f,
                VelocityX = npc.velocity.X / 16f,
                VelocityY = npc.velocity.Y / 16f,
                HealthFraction = npc.lifeMax > 0 ? npc.life / (float)npc.lifeMax : 0f,
                CurrentLife = npc.life,
                MaxLife = npc.lifeMax,
                Damage = npc.damage,
                Defense = npc.defense,
                LineOfSight = Collision.CanHitLine(Player.Center, 1, 1, npc.Center, 1, 1) ? 1f : 0f,
                Threat = Math.Max(npc.damage, 0) / 100f,
                Distance = (float)Math.Sqrt(enemyDistanceSquared) / 16f,
            });
            if (enemyDistanceSquared < closestEnemyDistanceSquared)
            {
                closestEnemyOffset = enemyOffset;
                closestEnemyDistanceSquared = enemyDistanceSquared;
                hasClosestEnemy = true;
            }
        }

        return new CombatObservation
        {
            Tick = (ulong)Main.GameUpdateCount,
            GridFlags = gridFlags,
            EnemyTypeIds = enemyTypeIds,
            PlayerFeatures = BuildPlayerFeatures(playerTile, closestEnemyOffset, hasClosestEnemy),
            WeaponFeatures = BuildWeaponFeatures(),
            WeaponTypeId = Player.HeldItem.type,
            WeaponName = Player.HeldItem.Name,
            Hotbar = BuildHotbarObservations(),
            AttackAttempted = attackAttemptedThisTick,
            AttackHit = attackHitThisTick,
            AttackNotReady = attackNotReadyThisTick,
            DamageTaken = Math.Max(previousLife - Player.statLife, 0),
            EnemiesKilled = enemiesKilledThisTick,
            WasAlive = wasAlive,
            IsAlive = !Player.dead,
            Enemies = enemies,
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

    private bool TryGetTargetDirection(int targetSlot, out Vector2 direction)
    {
        if (!TryGetTargetCenter(targetSlot, out Vector2 closestEnemyCenter))
        {
            direction = Vector2.UnitX * Player.direction;
            return false;
        }

        direction = closestEnemyCenter - Player.Center;
        if (direction == Vector2.Zero)
        {
            direction = Vector2.UnitX * Player.direction;
        }
        else
        {
            direction.Normalize();
        }
        return true;
    }

    private bool TryGetTargetCenter(int targetSlot, out Vector2 center)
    {
        List<NPC> enemies = new();
        for (int npcIndex = 0; npcIndex < Main.maxNPCs; npcIndex++)
        {
            NPC npc = Main.npc[npcIndex];
            if (!npc.active || !npc.CanBeChasedBy())
            {
                continue;
            }

            enemies.Add(npc);
        }

        enemies.Sort((left, right) =>
            Vector2.DistanceSquared(Player.Center, left.Center)
                .CompareTo(Vector2.DistanceSquared(Player.Center, right.Center)));
        if (enemies.Count == 0)
        {
            center = Player.Center + Vector2.UnitX * Player.direction;
            return false;
        }

        center = enemies[Math.Clamp(targetSlot, 0, enemies.Count - 1)].Center;
        return true;
    }

    private void SetVirtualCursor(Vector2 direction)
    {
        Vector2 cursorWorldPosition = Player.Center + direction * 1000f;
        Main.mouseX = (int)(cursorWorldPosition.X - Main.screenPosition.X);
        Main.mouseY = (int)(cursorWorldPosition.Y - Main.screenPosition.Y);
    }

    private float[] BuildPlayerFeatures(Point playerTile, Vector2 closestEnemyOffset, bool hasClosestEnemy)
    {
        Item heldItem = Player.HeldItem;
        bool canAttack = IsCombatItem(heldItem)
            && Player.itemTime <= 0
            && Player.itemAnimation <= 0
            && !Player.dead;
        float cooldownFraction = heldItem.useTime > 0
            ? Math.Clamp(Player.itemTime / (float)heldItem.useTime, 0f, 1f)
            : 0f;
        float animationFraction = heldItem.useAnimation > 0
            ? Math.Clamp(Player.itemAnimation / (float)heldItem.useAnimation, 0f, 1f)
            : 0f;
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
            closestEnemyOffset.X / (GridWidth * 8f),
            closestEnemyOffset.Y / (GridHeight * 8f),
            closestEnemyOffset.Length() / (GridWidth * 8f),
            hasClosestEnemy ? 1f : 0f,
            canAttack ? 1f : 0f,
            cooldownFraction,
            animationFraction,
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

    private List<HotbarSlotObservation> BuildHotbarObservations()
    {
        List<HotbarSlotObservation> hotbar = new();
        for (int slot = 0; slot < 10; slot++)
        {
            Item item = Player.inventory[slot];
            bool validWeapon = IsCombatItem(item);
            hotbar.Add(new HotbarSlotObservation
            {
                Slot = slot,
                ItemTypeId = item.type,
                ValidWeapon = validWeapon ? 1f : 0f,
                Damage = item.damage / 500f,
                UseTime = item.useTime / 100f,
                Knockback = item.knockBack / 20f,
                Shoot = item.shoot / 1000f,
                ShootSpeed = item.shootSpeed / 100f,
                Melee = item.DamageType.CountsAsClass(DamageClass.Melee) ? 1f : 0f,
                NoMelee = item.noMelee ? 1f : 0f,
                Range = item.shoot > 0 ? item.shootSpeed / 100f : item.width / 100f,
                Area = item.width * item.height / 10000f,
                Selected = Player.selectedItem == slot ? 1f : 0f,
            });
        }
        return hotbar;
    }
}
