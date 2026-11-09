using Terraria.ModLoader;
using Terraria.ModLoader.Config;

namespace TerrariaAgent;

public sealed class TerrariaAgentConfig : ModConfig
{
    public override ConfigScope Mode => ConfigScope.ClientSide;

    public bool PauseWhenUnfocused { get; set; } = true;
}
