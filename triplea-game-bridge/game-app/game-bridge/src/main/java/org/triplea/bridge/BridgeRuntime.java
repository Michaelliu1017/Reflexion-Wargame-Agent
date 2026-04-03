package org.triplea.bridge;

import static games.strategy.engine.framework.startup.mc.ClientModel.CLIENT_READY_CHANNEL;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import games.strategy.engine.data.GameStep;
import games.strategy.engine.framework.GameDataManager;
import games.strategy.engine.framework.GameObjectStreamFactory;
import games.strategy.engine.framework.startup.launcher.IServerReady;
import games.strategy.engine.framework.startup.mc.IClientChannel;
import games.strategy.engine.framework.startup.mc.IObserverWaitingToJoin;
import games.strategy.engine.framework.startup.mc.ServerModel;
import games.strategy.engine.framework.message.PlayerListing;
import games.strategy.engine.message.RemoteName;
import games.strategy.net.ClientMessengerFactory;
import games.strategy.net.IClientMessenger;
import games.strategy.net.INode;
import games.strategy.net.Messengers;
import games.strategy.net.websocket.ClientNetworkBridge;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.ThreadRunner;

/**
 * Connects to a TripleA host, takes the specified player (e.g. Japan), and exposes state/actions
 * for the HTTP API.
 */
@Slf4j
public final class BridgeRuntime {

  private final String host;
  private final int port;
  private final String playerName;
  private final String takePlayerName;

  private final GameObjectStreamFactory objectStreamFactory = new GameObjectStreamFactory(null);
  private final AtomicReference<BridgePlayer> bridgePlayerRef = new AtomicReference<>();
  private final BridgeLaunchAction launchAction = new BridgeLaunchAction();

  private IClientMessenger messenger;
  private Messengers messengers;
  private volatile boolean gameStarted;

  public BridgeRuntime(
      final String host,
      final int port,
      final String playerName,
      final String takePlayerName) {
    this.host = host;
    this.port = port;
    this.playerName = playerName;
    this.takePlayerName = takePlayerName;
  }

  /** Connects to the host, registers as client, and takes the target player (e.g. Japan). */
  public void connect() throws Exception {
    final var props =
        games.strategy.engine.framework.startup.mc.ClientModel.ClientProps.builder()
            .host(host)
            .port(port)
            .name(playerName)
            .build();
    messenger =
        ClientMessengerFactory.newClientMessenger(
            props, objectStreamFactory, new HeadlessLogin());
    messengers = new Messengers(messenger);

    final IClientChannel channelListener =
        new IClientChannel() {
          @Override
          public void playerListingChanged(final PlayerListing listing) {
            onPlayerListingChanged(listing);
          }

          @Override
          public void doneSelectingPlayers(final byte[] gameData, final Map<String, INode> players) {
            onDoneSelectingPlayers(gameData, players, false);
          }

          @Override
          public void gameReset() {
            objectStreamFactory.setData(null);
            gameStarted = false;
          }
        };

    final IObserverWaitingToJoin observerWaitingToJoin =
        new IObserverWaitingToJoin() {
          @Override
          public void joinGame(final byte[] gameData, final Map<String, INode> players) {
            onDoneSelectingPlayers(gameData, players, true);
          }

          @Override
          public void cannotJoinGame(final String reason) {
            log.warn("Cannot join game: {}", reason);
          }
        };

    messengers.registerChannelSubscriber(channelListener, IClientChannel.CHANNEL_NAME);
    final RemoteName observerName =
        new RemoteName(
            "games.strategy.engine.framework.startup.mc.ServerModel.OBSERVER"
                + messenger.getLocalNode().getName(),
            IObserverWaitingToJoin.class);
    messengers.registerRemote(observerWaitingToJoin, observerName);

    final var serverStartup =
        (games.strategy.engine.framework.startup.mc.IServerStartupRemote)
            messengers.getRemote(ServerModel.SERVER_REMOTE_NAME);
    final PlayerListing listing = serverStartup.getPlayerListing();
    onPlayerListingChanged(listing);

    if (!serverStartup.isGameStarted(messenger.getLocalNode())) {
      messengers.unregisterRemote(observerName);
      log.info("Taking player: {}", takePlayerName);
      serverStartup.takePlayer(messenger.getLocalNode(), takePlayerName);
    }
  }

  private void onPlayerListingChanged(final PlayerListing listing) {
    log.debug("Player listing changed: {}", listing);
  }

  private void onDoneSelectingPlayers(
      final byte[] gameDataBytes,
      final Map<String, INode> players,
      final boolean gameRunning) {
    log.info("Done selecting players, gameRunning={}", gameRunning);
    ThreadRunner.runInNewThread(
        () -> {
          try {
            startGame(gameDataBytes, players, gameRunning);
          } catch (Exception e) {
            log.error("Failed to start game", e);
          }
        });
  }

  private void startGame(
      final byte[] gameDataBytes,
      final Map<String, INode> players,
      final boolean gameRunning) {
    final var data =
        GameDataManager.loadGame(new ByteArrayInputStream(gameDataBytes)).orElse(null);
    if (data == null) {
      log.error("Failed to load game data");
      return;
    }
    objectStreamFactory.setData(data);

    final Map<String, games.strategy.engine.framework.startup.ui.PlayerTypes.Type> playerMapping =
        new HashMap<>();
    final INode localNode = messenger.getLocalNode();
    for (final Map.Entry<String, INode> e : players.entrySet()) {
      if (localNode.equals(e.getValue())) {
        playerMapping.put(e.getKey(), new BridgePlayerType(bridgePlayerRef));
      }
    }

    final var playerSet = data.getGameLoader().newPlayers(playerMapping);
    final var game =
        new games.strategy.engine.framework.ClientGame(
            data,
            playerSet,
            players,
            messengers,
            ClientNetworkBridge.NO_OP_SENDER);

    data.getGameLoader().startGame(game, playerSet, launchAction, null);
    gameStarted = true;
    log.info("Bridge game started, our player ref: {}", bridgePlayerRef.get());

    if (!gameRunning) {
      try {
        ((IServerReady) messengers.getRemote(CLIENT_READY_CHANNEL)).clientReady();
      } catch (Exception e) {
        log.warn("Failed to signal client ready", e);
      }
    }
  }

  /** Returns a JSON-serializable state snapshot for the LLM. */
  public Map<String, Object> getState() {
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) {
      final Map<String, Object> gameMap = new HashMap<>();
      gameMap.put("connected", false);
      gameMap.put("currentPlayerName", null);
      gameMap.put("controlledPlayerName", takePlayerName);
      return Map.of(
          "game", gameMap,
          "japan", Map.of("pus", 0, "techTokens", 0),
          "territories", List.<Map<String, Object>>of(),
          "unitsByTerritory", Map.<String, Object>of(),
          "purchaseOptions", List.of(),
          "placeOptions", List.of());
    }
    final var data = game.getData();
    final var bridgePlayer = bridgePlayerRef.get();
    final var placeDelegate =
        bridgePlayer != null ? bridgePlayer.getRemotePlaceDelegateIfPlaceStep() : null;
    try (var ignored = data.acquireReadLock()) {
      final String stepName = data.getSequence().getStep().getName();
      return BridgeStateBuilder.buildState(data, takePlayerName, stepName, placeDelegate);
    } catch (Exception e) {
      log.warn("Build state failed, returning minimal state", e);
      final String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
      final Map<String, Object> gameMap = new HashMap<>();
      gameMap.put("connected", true);
      gameMap.put("currentPlayerName", null);
      gameMap.put("stateError", msg);
      gameMap.put("controlledPlayerName", takePlayerName);
      return Map.of(
          "game", gameMap,
          "japan", Map.of("pus", 0, "techTokens", 0),
          "territories", List.<Map<String, Object>>of(),
          "unitsByTerritory", Map.<String, Object>of(),
          "purchaseOptions", List.of(),
          "placeOptions", List.of());
    }
  }

  /** Returns the list of legal action types for the current phase (Japan's turn only). */
  public List<Map<String, String>> getLegalActions() {
    final var actions = new ArrayList<Map<String, String>>();
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) return actions;
    final var data = game.getData();
    try (var ignored = data.acquireReadLock()) {
      final var step = data.getSequence().getStep();
      final var current = step.getPlayerId();
      if (current == null || !takePlayerName.equals(current.getName())) {
        return actions;
      }
      final String stepName = step.getName();
      if (GameStep.isPurchaseStepName(stepName)) {
        actions.add(Map.of("type", "BUY_UNITS"));
      } else if (GameStep.isPlaceStepName(stepName)) {
        actions.add(Map.of("type", "PLACE_UNITS"));
      } else if (GameStep.isCombatMoveStepName(stepName)) {
        actions.add(Map.of("type", "PERFORM_MOVE", "phase", "combat"));
      } else if (GameStep.isNonCombatMoveStepName(stepName)) {
        actions.add(Map.of("type", "PERFORM_MOVE", "phase", "noncombat"));
      } else if (GameStep.isBattleStepName(stepName)) {
        actions.add(Map.of("type", "BATTLE", "note", "战斗伤亡已自动选择；可 END_TURN 进入下一场或下一阶段"));
      }
      actions.add(Map.of("type", "END_TURN"));
    }
    return actions;
  }

  /**
   * Applies an action. Returns { "ok": true } or { "ok": false, "error": "...", "details": {...} }.
   */
  public Map<String, Object> applyAction(final JsonObject action) {
    final String type = action.has("type") ? action.get("type").getAsString() : null;
    if ("END_TURN".equals(type)
        || "COMBAT_MOVE".equals(type)
        || "NON_COMBAT_MOVE".equals(type)
        || "BATTLE".equals(type)) {
      return applyEndTurn();
    }
    if ("BUY_UNITS".equals(type)) {
      return applyBuyUnits(action);
    }
    if ("PLACE_UNITS".equals(type)) {
      return applyPlaceUnits(action);
    }
    if ("PERFORM_MOVE".equals(type)) {
      return applyPerformMove(action);
    }
    return Map.of("ok", false, "error", "Unsupported action type: " + type);
  }

  private Map<String, Object> applyPerformMove(final JsonObject action) {
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) {
      return Map.of("ok", false, "error", "Game not started");
    }
    final var data = game.getData();
    try (var ignored = data.acquireReadLock()) {
      final var step = data.getSequence().getStep();
      final String stepName = step.getName();
      if (!GameStep.isCombatMoveStepName(stepName) && !GameStep.isNonCombatMoveStepName(stepName)) {
        return Map.of("ok", false, "error", "Not in move phase");
      }
      if (step.getPlayerId() == null || !takePlayerName.equals(step.getPlayerId().getName())) {
        return Map.of("ok", false, "error", "Not " + takePlayerName + "'s turn");
      }
    }
    final String from = action.has("from") ? action.get("from").getAsString() : null;
    final String to = action.has("to") ? action.get("to").getAsString() : null;
    if (from == null || to == null) {
      return Map.of("ok", false, "error", "PERFORM_MOVE requires 'from' and 'to' territories");
    }
    final List<BridgePlayer.MoveSpec> specs = new ArrayList<>();
    if (action.has("units") && action.get("units").isJsonArray()) {
      for (JsonElement el : action.getAsJsonArray("units")) {
        if (!el.isJsonObject()) continue;
        final JsonObject o = el.getAsJsonObject();
        final String unitType = o.has("unitType") ? o.get("unitType").getAsString() : null;
        final int count = o.has("count") ? o.get("count").getAsInt() : 0;
        if (unitType != null && count > 0) {
          specs.add(new BridgePlayer.MoveSpec(from, to, unitType, count));
        }
      }
    }
    if (specs.isEmpty()) {
      return Map.of("ok", false, "error", "PERFORM_MOVE requires at least one unit in 'units'");
    }
    final BridgePlayer player = bridgePlayerRef.get();
    if (player == null) {
      return Map.of("ok", false, "error", "Bridge player not found");
    }
    final String err = player.performMove(from, to, specs);
    if (err != null) {
      return Map.of("ok", false, "error", err);
    }
    return Map.of("ok", true);
  }

  private Map<String, Object> applyBuyUnits(final JsonObject action) {
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) {
      return Map.of("ok", false, "error", "Game not started");
    }
    final var data = game.getData();
    try (var ignored = data.acquireReadLock()) {
      final var step = data.getSequence().getStep();
      if (!GameStep.isPurchaseStepName(step.getName())) {
        return Map.of("ok", false, "error", "Not in purchase phase");
      }
      if (step.getPlayerId() == null || !takePlayerName.equals(step.getPlayerId().getName())) {
        return Map.of("ok", false, "error", "Not " + takePlayerName + "'s turn");
      }
    }
    final BridgePlayer player = bridgePlayerRef.get();
    if (player == null) {
      return Map.of("ok", false, "error", "Bridge player not found");
    }
    final Map<String, Integer> items = new HashMap<>();
    if (action.has("items") && action.get("items").isJsonArray()) {
      for (JsonElement el : action.getAsJsonArray("items")) {
        if (!el.isJsonObject()) continue;
        final JsonObject o = el.getAsJsonObject();
        final String unitType = o.has("unitType") ? o.get("unitType").getAsString() : null;
        final int count = o.has("count") ? o.get("count").getAsInt() : 0;
        if (unitType != null && count > 0) {
          items.merge(unitType, count, Integer::sum);
        }
      }
    }
    if (items.isEmpty()) {
      return Map.of("ok", false, "error", "No items in BUY_UNITS");
    }
    final String err = player.performPurchase(items);
    if (err != null) {
      return Map.of("ok", false, "error", err);
    }
    return Map.of("ok", true);
  }

  private Map<String, Object> applyPlaceUnits(final JsonObject action) {
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) {
      return Map.of("ok", false, "error", "Game not started");
    }
    final var data = game.getData();
    try (var ignored = data.acquireReadLock()) {
      final var step = data.getSequence().getStep();
      if (!GameStep.isPlaceStepName(step.getName())) {
        return Map.of("ok", false, "error", "Not in place phase");
      }
      if (step.getPlayerId() == null || !takePlayerName.equals(step.getPlayerId().getName())) {
        return Map.of("ok", false, "error", "Not " + takePlayerName + "'s turn");
      }
    }
    final BridgePlayer player = bridgePlayerRef.get();
    if (player == null) {
      return Map.of("ok", false, "error", "Bridge player not found");
    }
    final List<BridgePlayer.PlacementSpec> specs = new ArrayList<>();
    if (action.has("placements") && action.get("placements").isJsonArray()) {
      for (JsonElement el : action.getAsJsonArray("placements")) {
        if (!el.isJsonObject()) continue;
        final JsonObject o = el.getAsJsonObject();
        final String territory = o.has("territory") ? o.get("territory").getAsString() : null;
        final String unitType = o.has("unitType") ? o.get("unitType").getAsString() : null;
        final int count = o.has("count") ? o.get("count").getAsInt() : 0;
        if (territory != null && unitType != null && count > 0) {
          specs.add(new BridgePlayer.PlacementSpec(territory, unitType, count));
        }
      }
    }
    if (specs.isEmpty()) {
      return Map.of("ok", false, "error", "No placements in PLACE_UNITS");
    }
    final String err = player.performPlace(specs);
    if (err != null) {
      return Map.of("ok", false, "error", err);
    }
    return Map.of("ok", true);
  }

  private Map<String, Object> applyEndTurn() {
    final games.strategy.engine.framework.IGame game = launchAction.getGame();
    if (game == null || !gameStarted) {
      return Map.of("ok", false, "error", "Game not started");
    }
    final var data = game.getData();
    try (var ignored = data.acquireReadLock()) {
      final var step = data.getSequence().getStep();
      final var active = step.getPlayerId();
      if (active == null || !takePlayerName.equals(active.getName())) {
        return Map.of(
            "ok", false,
            "error", "Not " + takePlayerName + "'s turn (current: " + (active != null ? active.getName() : "null") + ")");
      }
    }
    final BridgePlayer player = bridgePlayerRef.get();
    if (player == null) {
      return Map.of("ok", false, "error", "Bridge player not found");
    }
    player.signalStepDone();
    return Map.of("ok", true);
  }

  @Nullable
  public BridgePlayer getBridgePlayer() {
    return bridgePlayerRef.get();
  }

  public boolean isGameStarted() {
    return gameStarted;
  }

  public void shutdown() {
    if (messenger != null) {
      messenger.shutDown();
    }
  }
}
