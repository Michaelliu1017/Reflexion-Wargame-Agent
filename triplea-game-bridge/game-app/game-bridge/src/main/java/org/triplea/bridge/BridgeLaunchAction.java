package org.triplea.bridge;

import games.strategy.engine.chat.HeadlessChat;
import games.strategy.engine.chat.MessengersChatTransmitter;
import games.strategy.engine.framework.AutoSaveFileUtils;
import games.strategy.engine.framework.ServerGame;
import games.strategy.engine.framework.startup.WatcherThreadMessaging;
import games.strategy.engine.framework.startup.mc.IServerStartupRemote;
import games.strategy.engine.framework.startup.mc.ServerConnectionProps;
import games.strategy.engine.framework.startup.mc.ServerModel;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.engine.player.Player;
import games.strategy.net.Messengers;
import games.strategy.net.websocket.ClientNetworkBridge;
import java.nio.file.Path;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.triplea.game.chat.ChatModel;

/** Minimal LaunchAction for the bridge client: no UI, just stores game reference. */
@Slf4j
public final class BridgeLaunchAction implements games.strategy.engine.framework.startup.launcher.LaunchAction {

  private volatile games.strategy.engine.framework.IGame game;

  games.strategy.engine.framework.IGame getGame() {
    return game;
  }

  @Override
  public void startGame(
      final games.strategy.engine.framework.LocalPlayers localPlayers,
      final games.strategy.engine.framework.IGame game,
      final Set<Player> players,
      final games.strategy.engine.chat.Chat chat) {
    this.game = game;
    log.info("Bridge launch action: game started");
  }

  @Override
  public void handleGameInterruption(
      final GameSelectorModel gameSelectorModel, final ServerModel serverModel) {
    log.info("Bridge: handleGameInterruption");
  }

  @Override
  public void onGameInterrupt() {
    log.info("Bridge: onGameInterrupt");
  }

  @Override
  public void onEnd(final String message) {
    log.info("Bridge: onEnd - {}", message);
  }

  @Override
  public Collection<PlayerTypes.Type> getPlayerTypes() {
    return List.of();
  }

  @Override
  public Path getAutoSaveFile() {
    return Path.of("bridge-autosave.tsvg");
  }

  @Override
  public void onLaunch(final ServerGame serverGame) {}

  @Override
  public AutoSaveFileUtils getAutoSaveFileUtils() {
    return new AutoSaveFileUtils();
  }

  @Override
  public ChatModel createChatModel(
      final String chatName,
      final Messengers messengers,
      final ClientNetworkBridge clientNetworkBridge) {
    return new HeadlessChat(
        new games.strategy.engine.chat.Chat(
            new MessengersChatTransmitter(chatName, messengers, clientNetworkBridge)));
  }

  @Override
  public boolean shouldMinimizeExpensiveAiUse() {
    return true;
  }

  @Override
  public WatcherThreadMessaging createThreadMessaging() {
    return new WatcherThreadMessaging.HeadlessWatcherThreadMessaging();
  }

  @Override
  public Optional<ServerConnectionProps> getFallbackConnection(final Runnable cancelAction) {
    return Optional.empty();
  }

  @Override
  public void handleError(final String error) {
    log.warn("Bridge: handleError - {}", error);
  }

  @Override
  public IServerStartupRemote getStartupRemote(final IServerStartupRemote.ServerModelView serverModelView) {
    throw new UnsupportedOperationException("Bridge client does not provide server startup remote");
  }

  @Override
  public boolean promptGameStop(final String status, final String title, @Nullable final Path mapLocation) {
    return true;
  }

  @Override
  public PlayerTypes.Type getDefaultLocalPlayerType() {
    return PlayerTypes.PRO_AI;
  }
}
