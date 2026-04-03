package org.triplea.bridge;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.GameStep;
import games.strategy.engine.data.ProductionRule;
import games.strategy.engine.data.Resource;
import games.strategy.engine.data.Territory;
import games.strategy.engine.data.Unit;
import games.strategy.engine.data.UnitType;
import games.strategy.triplea.Constants;
import games.strategy.triplea.attachments.TerritoryAttachment;
import games.strategy.triplea.delegate.remote.IAbstractPlaceDelegate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Builds JSON-serializable state and option maps for the bridge API. */
public final class BridgeStateBuilder {

  private BridgeStateBuilder() {}

  public static Map<String, Object> buildState(
      final GameData data,
      final String japanName,
      final String stepName,
      @Nullable final IAbstractPlaceDelegate placeDelegate) {
    final var game = new HashMap<String, Object>();
    final var seq = data.getSequence();
    final var step = seq.getStep();
    game.put("round", seq.getRound());
    game.put("stepName", stepName);
    game.put("phaseName", step.getDisplayName() != null ? step.getDisplayName() : stepName);
    game.put("currentPlayerName", step.getPlayerId() != null ? step.getPlayerId().getName() : null);
    game.put("controlledPlayerName", japanName);

    final GamePlayer japan = data.getPlayerList().getPlayerId(japanName);
    final var japanMap = new HashMap<String, Object>();
    japanMap.put("pus", japan != null ? getPus(japan, data) : 0);
    japanMap.put("techTokens", japan != null ? getTechTokens(japan, data) : 0);

    final List<Map<String, Object>> territories = new ArrayList<>();
    for (final Territory t : data.getMap().getTerritories()) {
      final var tm = new HashMap<String, Object>();
      tm.put("name", t.getName());
      tm.put("owner", t.getOwner() != null ? t.getOwner().getName() : null);
      tm.put("isLand", !t.isWater());
      tm.put("isWater", t.isWater());
      tm.put("neighbors",
          data.getMap().getNeighbors(t).stream().map(Territory::getName).toList());
      tm.put("unitsSummary", unitsSummary(t.getUnitCollection()));
      final int prod = TerritoryAttachment.getProduction(t);
      if (prod != 0) tm.put("puValue", prod);
      territories.add(tm);
    }

    final Map<String, Map<String, Integer>> unitsByTerritory = new HashMap<>();
    if (japan != null) {
      for (final Territory t : data.getMap().getTerritories()) {
        final Map<String, Integer> counts = new HashMap<>();
        for (final Unit u : t.getUnitCollection().getUnits()) {
          if (u.getOwner().equals(japan)) {
            final String type = u.getType().getName();
            counts.put(type, counts.getOrDefault(type, 0) + 1);
          }
        }
        if (!counts.isEmpty()) unitsByTerritory.put(t.getName(), counts);
      }
    }

    List<Map<String, Object>> purchaseOptions = List.of();
    if (japan != null
        && GameStep.isPurchaseStepName(stepName)
        && japan.getName().equals(step.getPlayerId() != null ? step.getPlayerId().getName() : null)
        && japan.getProductionFrontier() != null) {
      final Resource pus = data.getResourceList().getResourceOrThrow(Constants.PUS);
      final int available = japan.getResources().getQuantity(pus);
      purchaseOptions = new ArrayList<>();
      for (final ProductionRule rule : japan.getProductionFrontier().getRules()) {
        for (final var key : rule.getResults().keySet()) {
          if (key instanceof UnitType) {
            final int cost = rule.getCosts().getInt(pus);
            final int maxAffordable = cost > 0 ? available / cost : 0;
            final var om = new HashMap<String, Object>();
            om.put("unitType", ((UnitType) key).getName());
            om.put("cost", cost);
            om.put("maxAffordable", maxAffordable);
            purchaseOptions.add(om);
            break;
          }
        }
      }
    }

    List<Map<String, Object>> placeOptions = List.of();
    if (japan != null
        && placeDelegate != null
        && GameStep.isPlaceStepName(stepName)
        && japan.getName().equals(step.getPlayerId() != null ? step.getPlayerId().getName() : null)
        && !japan.getUnitCollection().isEmpty()) {
      final List<Map<String, Object>> options = new ArrayList<>();
      for (final Territory t : data.getMap().getTerritories()) {
        if (!t.isOwnedBy(japan)) continue;
        try {
          final var pu = placeDelegate.getPlaceableUnits(japan.getUnitCollection(), t);
          if (pu.getErrorMessage() != null) continue;
          final var pm = new HashMap<String, Object>();
          pm.put("territory", t.getName());
          pm.put("maxPlaceCapacity", pu.getMaxUnits() == -1 ? null : pu.getMaxUnits());
          options.add(pm);
        } catch (Exception ignored) {
          // Skip this territory if RPC fails (e.g. host busy or serialization)
        }
      }
      placeOptions = options;
    }

    final var out = new HashMap<String, Object>();
    out.put("game", game);
    out.put("japan", japanMap);
    out.put("territories", territories);
    out.put("unitsByTerritory", unitsByTerritory);
    out.put("purchaseOptions", purchaseOptions);
    out.put("placeOptions", placeOptions);
    return out;
  }

  private static int getPus(final GamePlayer player, final GameData data) {
    final Optional<Resource> r = data.getResourceList().getResourceOptional(Constants.PUS);
    return r.map(res -> player.getResources().getQuantity(res)).orElse(0);
  }

  private static int getTechTokens(final GamePlayer player, final GameData data) {
    final Optional<Resource> r =
        data.getResourceList().getResourceOptional(Constants.TECH_TOKENS);
    return r.map(res -> player.getResources().getQuantity(res)).orElse(0);
  }

  private static Map<String, Integer> unitsSummary(
      final games.strategy.engine.data.UnitCollection coll) {
    return coll.stream()
        .collect(Collectors.groupingBy(u -> u.getType().getName(), Collectors.summingInt(x -> 1)));
  }
}
