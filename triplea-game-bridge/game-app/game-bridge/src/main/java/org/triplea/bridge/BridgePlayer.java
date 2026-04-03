package org.triplea.bridge;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.MoveDescription;
import games.strategy.engine.data.ProductionRule;
import games.strategy.engine.data.Resource;
import games.strategy.engine.data.Route;
import games.strategy.engine.data.Territory;
import games.strategy.engine.data.Unit;
import games.strategy.engine.data.UnitType;
import games.strategy.triplea.delegate.data.CasualtyDetails;
import games.strategy.triplea.delegate.data.CasualtyList;
import games.strategy.engine.data.GameStep;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.player.Player;
import games.strategy.engine.player.PlayerBridge;
import games.strategy.triplea.Constants;
import games.strategy.triplea.delegate.DiceRoll;
import games.strategy.triplea.delegate.remote.IAbstractPlaceDelegate;
import games.strategy.triplea.delegate.remote.IMoveDelegate;
import games.strategy.triplea.delegate.remote.IPurchaseDelegate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.collections.IntegerMap;
import org.triplea.util.Tuple;

/**
 * Player implementation for the Bridge: blocks in {@link #start(String)} until {@link
 * #signalStepDone()} is called (e.g. when POST /act with END_TURN is received).
 */
@Slf4j
public final class BridgePlayer implements Player {

  private static final String LABEL = "Bridge";

  private final String name;
  private final ReentrantLock stepLock = new ReentrantLock();
  private final Condition stepDone = stepLock.newCondition();
  private volatile String currentStepName;

  private PlayerBridge playerBridge;
  private GamePlayer gamePlayer;

  BridgePlayer(final String name) {
    this.name = name;
  }

  @Override
  public void initialize(final PlayerBridge bridge, final GamePlayer gamePlayer) {
    this.playerBridge = bridge;
    this.gamePlayer = gamePlayer;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public String getPlayerLabel() {
    return LABEL;
  }

  @Override
  public boolean isAi() {
    return false;
  }

  @Override
  public void start(final String stepName) {
    currentStepName = stepName;
    log.info("Bridge player {} entered step: {}", name, stepName);
    stepLock.lock();
    try {
      stepDone.await();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      log.warn("Bridge player {} interrupted waiting for step done", name);
    } finally {
      stepLock.unlock();
    }
    log.info("Bridge player {} leaving step: {}", name, stepName);
  }

  /** Called when the bridge receives an END_TURN action so that {@link #start(String)} returns. */
  void signalStepDone() {
    stepLock.lock();
    try {
      stepDone.signal();
    } finally {
      stepLock.unlock();
    }
  }

  @Override
  public void stopGame() {
    signalStepDone();
  }

  @Override
  public GamePlayer getGamePlayer() {
    return gamePlayer;
  }

  /**
   * Performs a purchase via the server delegate. Called from HTTP thread. Returns error message or
   * null on success.
   */
  @Nullable
  public String performPurchase(final Map<String, Integer> unitTypeToCount) {
    if (playerBridge == null || gamePlayer == null) {
      return "Bridge not initialized";
    }
    final GameData data = playerBridge.getGameData();
    if (gamePlayer.getProductionFrontier() == null) {
      return "No production frontier";
    }
    final IntegerMap<ProductionRule> purchase = new IntegerMap<>();
    try (GameData.Unlocker ignored = data.acquireReadLock()) {
      final Resource pus = data.getResourceList().getResourceOrThrow(Constants.PUS);
      final int available = gamePlayer.getResources().getQuantity(pus);
      int totalCost = 0;
      for (final Map.Entry<String, Integer> e : unitTypeToCount.entrySet()) {
        final String typeName = e.getKey();
        final int count = e.getValue();
        if (count <= 0) continue;
        final ProductionRule rule = findProductionRule(gamePlayer, typeName);
        if (rule == null) {
          return "Unknown unit type: " + typeName;
        }
        final int costPer = rule.getCosts().getInt(pus);
        totalCost += costPer * count;
        if (totalCost > available) {
          return "Not enough PUs (need " + totalCost + ", have " + available + ")";
        }
        purchase.add(rule, count);
      }
    }
    try {
      final IPurchaseDelegate delegate =
          (IPurchaseDelegate) playerBridge.getRemoteDelegate();
      final String err = delegate.purchase(purchase);
      return err;
    } catch (Exception e) {
      log.warn("Purchase failed", e);
      return e.getMessage();
    }
  }

  /**
   * Performs placements via the server delegate. Each placement is (territoryName, unitTypeName,
   * count). Returns error message or null on success.
   */
  @Nullable
  public String performPlace(final List<PlacementSpec> placements) {
    if (playerBridge == null || gamePlayer == null) {
      return "Bridge not initialized";
    }
    final GameData data = playerBridge.getGameData();
    try {
      final IAbstractPlaceDelegate delegate =
          (IAbstractPlaceDelegate) playerBridge.getRemoteDelegate();
      for (final PlacementSpec spec : placements) {
        final Territory territory = data.getMap().getTerritoryOrNull(spec.territory);
        if (territory == null) {
          return "Unknown territory: " + spec.territory;
        }
        final List<Unit> toPlace = new ArrayList<>();
        for (final Unit u : gamePlayer.getUnitCollection()) {
          if (toPlace.size() >= spec.count) break;
          if (u.getType().getName().equalsIgnoreCase(spec.unitType)) {
            toPlace.add(u);
          }
        }
        if (toPlace.size() < spec.count) {
          return "Not enough " + spec.unitType + " to place (need " + spec.count + ", have " + toPlace.size() + ")";
        }
        final Optional<String> err =
            delegate.placeUnits(toPlace, territory, IAbstractPlaceDelegate.BidMode.NOT_BID);
        if (err.isPresent()) {
          return err.get();
        }
      }
      return null;
    } catch (Exception e) {
      log.warn("Place failed", e);
      return e.getMessage();
    }
  }

  private static ProductionRule findProductionRule(final GamePlayer player, final String unitTypeName) {
    if (player.getProductionFrontier() == null) return null;
    for (final ProductionRule rule : player.getProductionFrontier().getRules()) {
      for (final var key : rule.getResults().keySet()) {
        if (key instanceof UnitType && ((UnitType) key).getName().equalsIgnoreCase(unitTypeName)) {
          return rule;
        }
      }
    }
    return null;
  }

  /**
   * Returns the remote place delegate when the current step is a Place step (for building
   * placeOptions in state). Returns null otherwise.
   */
  @Nullable
  public IAbstractPlaceDelegate getRemotePlaceDelegateIfPlaceStep() {
    if (playerBridge == null) return null;
    final GameData data = playerBridge.getGameData();
    try (GameData.Unlocker ignored = data.acquireReadLock()) {
      final String stepName = data.getSequence().getStep().getName();
      if (stepName == null || !games.strategy.engine.data.GameStep.isPlaceStepName(stepName)) {
        return null;
      }
      return (IAbstractPlaceDelegate) playerBridge.getRemoteDelegate();
    }
  }

  /** One placement: territory name, unit type name, count. */
  public static final class PlacementSpec {
    public final String territory;
    public final String unitType;
    public final int count;

    public PlacementSpec(final String territory, final String unitType, final int count) {
      this.territory = territory;
      this.unitType = unitType;
      this.count = count;
    }
  }

  /** One move: from territory, to territory, unit type and count. */
  public static final class MoveSpec {
    public final String from;
    public final String to;
    public final String unitType;
    public final int count;

    public MoveSpec(final String from, final String to, final String unitType, final int count) {
      this.from = from;
      this.to = to;
      this.unitType = unitType;
      this.count = count;
    }
  }

  /**
   * Performs one move via the server MoveDelegate. Called from HTTP thread when in Combat Move or
   * Non-Combat Move step. Returns error message or null on success.
   */
  @Nullable
  public String performMove(final String fromTerritory, final String toTerritory,
      final List<MoveSpec> unitSpecs) {
    if (playerBridge == null || gamePlayer == null) {
      return "Bridge not initialized";
    }
    final GameData data = playerBridge.getGameData();
    final MoveDescription move;
    // Build move description inside read lock, then release lock before calling delegate
    try (GameData.Unlocker ignored = data.acquireReadLock()) {
      final Territory from = data.getMap().getTerritoryOrNull(fromTerritory);
      final Territory to = data.getMap().getTerritoryOrNull(toTerritory);
      if (from == null) return "Unknown territory: " + fromTerritory;
      if (to == null) return "Unknown territory: " + toTerritory;
      final List<Unit> unitsToMove = new ArrayList<>();
      for (final MoveSpec spec : unitSpecs) {
        if (spec.count <= 0) continue;
        int added = 0;
        for (final Unit u : from.getUnitCollection().getUnits()) {
          if (added >= spec.count) break;
          if (!u.getOwner().equals(gamePlayer)) continue;
          if (!u.getType().getName().equalsIgnoreCase(spec.unitType)) continue;
          if (unitsToMove.contains(u)) continue;
          unitsToMove.add(u);
          added++;
        }
        if (added < spec.count) {
          return "Not enough " + spec.unitType + " in " + fromTerritory + " (need " + spec.count + ")";
        }
      }
      if (unitsToMove.isEmpty()) {
        return "No units specified to move";
      }
      final Optional<Route> routeOpt =
          data.getMap().getRouteForUnits(from, to, t -> true, unitsToMove, gamePlayer);
      if (routeOpt.isEmpty()) {
        return "No valid route from " + fromTerritory + " to " + toTerritory;
      }

      // Build transportMap and dependentUnits for transport operations.
      // TripleA's MoveValidator requires explicit unit-to-transport assignment.
      final Map<Unit, Unit> transportMap = new HashMap<>();
      final Map<Unit, Collection<Unit>> dependentUnits = new HashMap<>();

      if (to.isWater() && !from.isWater()) {
        // ── Case 1: LOADING (land → sea) ─────────────────────────────────────
        // Infantry moves from a coastal territory onto a transport in the sea zone.
        // Find friendly transports in the DESTINATION sea zone.
        final List<Unit> availableTransports = new ArrayList<>();
        for (final Unit u : to.getUnitCollection().getUnits()) {
          if (!u.getOwner().equals(gamePlayer)) continue;
          if (u.getUnitAttachment().getTransportCapacity() > 0) {
            availableTransports.add(u);
          }
        }
        final Map<Unit, Integer> remainingCapacity = new HashMap<>();
        for (final Unit t : availableTransports) {
          remainingCapacity.put(t, t.getUnitAttachment().getTransportCapacity());
        }
        for (final Unit unit : unitsToMove) {
          final var ua = unit.getUnitAttachment();
          if (ua.isSea() || ua.isAir()) continue;
          final int cost = ua.getTransportCost();
          if (cost <= 0) continue;
          for (final Unit transport : availableTransports) {
            final int cap = remainingCapacity.getOrDefault(transport, 0);
            if (cap >= cost) {
              transportMap.put(unit, transport);
              remainingCapacity.put(transport, cap - cost);
              break;
            }
          }
        }

      } else if (to.isWater() && from.isWater()) {
        // ── Case 2: TRANSPORT MOVEMENT (sea → sea) ───────────────────────────
        // Transport moves between sea zones; loaded units travel as dependents.
        // Identify transports and land units within the units being moved.
        final List<Unit> movingTransports = new ArrayList<>();
        final List<Unit> movingLandUnits = new ArrayList<>();
        for (final Unit unit : unitsToMove) {
          final var ua = unit.getUnitAttachment();
          if (ua.getTransportCapacity() > 0) {
            movingTransports.add(unit);
            dependentUnits.put(unit, new ArrayList<>());
          } else if (!ua.isSea() && !ua.isAir()) {
            movingLandUnits.add(unit);
          }
        }
        final Map<Unit, Integer> remainingCapacity = new HashMap<>();
        for (final Unit t : movingTransports) {
          remainingCapacity.put(t, t.getUnitAttachment().getTransportCapacity());
        }
        for (final Unit unit : movingLandUnits) {
          final int cost = unit.getUnitAttachment().getTransportCost();
          if (cost <= 0) continue;
          for (final Unit transport : movingTransports) {
            final int cap = remainingCapacity.getOrDefault(transport, 0);
            if (cap >= cost) {
              transportMap.put(unit, transport);
              remainingCapacity.put(transport, cap - cost);
              ((List<Unit>) dependentUnits.get(transport)).add(unit);
              break;
            }
          }
        }

      }
      // ── Case 3: UNLOADING (sea → land) ─────────────────────────────────────
      // Infantry moves from a sea zone onto a coastal territory.
      // No transportMap needed; TripleA resolves the transport automatically.

      move = new MoveDescription(unitsToMove, routeOpt.get(), transportMap, dependentUnits);
    }
    // Call delegate OUTSIDE the read lock to avoid deadlock on shared GameData
    try {
      final IMoveDelegate delegate = (IMoveDelegate) playerBridge.getRemoteDelegate();
      final Optional<String> err = delegate.performMove(move);
      return err.orElse(null);
    } catch (Exception e) {
      log.warn("Perform move failed", e);
      return e.getMessage();
    }
  }

  // --- Stub implementations (not used in MVP) ---

  @Override
  public CasualtyDetails selectCasualties(
      final Collection<Unit> selectFrom,
      final Map<Unit, Collection<Unit>> dependents,
      final int count,
      final String message,
      final DiceRoll dice,
      final GamePlayer hit,
      final Collection<Unit> friendlyUnits,
      final Collection<Unit> enemyUnits,
      final boolean amphibious,
      final Collection<Unit> amphibiousLandAttackers,
      final CasualtyList defaultCasualties,
      final UUID battleId,
      final Territory battlesite,
      final boolean allowMultipleHitsPerUnit) {
    // Bridge 无 UI 选伤亡，直接采用引擎给出的默认选择，避免 Host 收到 null 导致 NPE
    return new CasualtyDetails(defaultCasualties != null ? defaultCasualties : new CasualtyDetails(), false);
  }

  @Override
  public int[] selectFixedDice(
      final int numDice, final int hitAt, final String title, final int diceSides) {
    return new int[0];
  }

  @Override
  public Territory selectBombardingTerritory(
      final Unit unit,
      final Territory unitTerritory,
      final Collection<Territory> territories,
      final boolean noneAvailable) {
    return null;
  }

  @Override
  public boolean selectAttackSubs(final Territory unitTerritory) {
    return false;
  }

  @Override
  public boolean selectAttackTransports(final Territory unitTerritory) {
    return false;
  }

  @Override
  public boolean selectAttackUnits(final Territory unitTerritory) {
    return false;
  }

  @Override
  public boolean selectShoreBombard(final Territory unitTerritory) {
    return false;
  }

  @Override
  public void reportError(final String error) {
    log.warn("Bridge player {} error: {}", name, error);
  }

  @Override
  public void reportMessage(final String message, final String title) {
    log.info("Bridge player {} message [{}]: {}", name, title, message);
  }

  @Override
  public boolean shouldBomberBomb(final Territory territory) {
    return false;
  }

  @Override
  public Unit whatShouldBomberBomb(
      final Territory territory,
      final Collection<Unit> potentialTargets,
      final Collection<Unit> bombers) {
    return null;
  }

  @Override
  public Territory whereShouldRocketsAttack(final Collection<Territory> candidates, final Territory from) {
    return null;
  }

  @Override
  public Collection<Unit> getNumberOfFightersToMoveToNewCarrier(
      final Collection<Unit> fightersThatCanBeMoved, final Territory from) {
    return List.of();
  }

  @Override
  public Territory selectTerritoryForAirToLand(
      final Collection<Territory> candidates,
      final Territory currentTerritory,
      final String unitMessage) {
    return candidates.isEmpty() ? null : candidates.iterator().next();
  }

  @Override
  public boolean confirmMoveInFaceOfAa(final Collection<Territory> aaFiringTerritories) {
    return true;
  }

  @Override
  public boolean confirmMoveKamikaze() {
    return false;
  }

  @Override
  public Optional<Territory> retreatQuery(
      final UUID battleId,
      final boolean submerge,
      final Territory battleTerritory,
      final Collection<Territory> possibleTerritories,
      final String message) {
    return Optional.empty();
  }

  @Override
  public Map<Territory, Collection<Unit>> scrambleUnitsQuery(
      final Territory scrambleTo,
      final Map<Territory, Tuple<Collection<Unit>, Collection<Unit>>> possibleScramblers) {
    return Map.of();
  }

  @Override
  public Collection<Unit> selectUnitsQuery(
      final Territory current, final Collection<Unit> possible, final String message) {
    return List.of();
  }

  @Override
  public void confirmEnemyCasualties(final UUID battleId, final String message, final GamePlayer hitPlayer) {}

  @Override
  public void confirmOwnCasualties(final UUID battleId, final String message) {}

  @Override
  public boolean acceptAction(
      final GamePlayer playerSendingProposal,
      final String acceptanceQuestion,
      final boolean politics) {
    return true;
  }

  @Override
  @Nullable
  public Map<Territory, Map<Unit, IntegerMap<Resource>>> selectKamikazeSuicideAttacks(
      final Map<Territory, Collection<Unit>> possibleUnitsToAttack) {
    return null;
  }

  @Override
  public Tuple<Territory, Set<Unit>> pickTerritoryAndUnits(
      final List<Territory> territoryChoices,
      final List<Unit> unitChoices,
      final int unitsPerPick) {
    return Tuple.of(territoryChoices.isEmpty() ? null : territoryChoices.get(0), Set.of());
  }
}
