package org.triplea.bridge;

import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.player.Player;
import java.util.concurrent.atomic.AtomicReference;

/** Player type that creates a {@link BridgePlayer} and registers it for the bridge to signal. */
public final class BridgePlayerType extends PlayerTypes.Type {

  private static final String LABEL = "Bridge";

  private final AtomicReference<BridgePlayer> playerRef;

  public BridgePlayerType(final AtomicReference<BridgePlayer> playerRef) {
    super(LABEL);
    this.playerRef = playerRef;
  }

  @Override
  public Player newPlayerWithName(final String name) {
    final BridgePlayer player = new BridgePlayer(name);
    playerRef.set(player);
    return player;
  }
}
