package org.triplea.bridge;

import games.strategy.engine.framework.startup.login.ClientLogin;
import java.util.HashMap;
import java.util.Map;
import org.triplea.config.product.ProductVersionReader;

/**
 * Headless connection login: returns engine version only, no password prompt. Use when connecting
 * to a host that does not require a password (e.g. local LAN host).
 */
public final class HeadlessLogin extends ClientLogin {

  private static final String ENGINE_VERSION_PROPERTY = "Engine.Version";

  public HeadlessLogin() {
    super(null, ProductVersionReader.getCurrentVersion());
  }

  @Override
  public Map<String, String> getProperties(final Map<String, String> challenge) {
    final Map<String, String> response = new HashMap<>();
    // Never prompt for password; only send a sanitized engine version (host must not require
    // password).
    var versionString = ProductVersionReader.getCurrentVersion().toString();
    final int plusIndex = versionString.indexOf('+');
    if (plusIndex >= 0) {
      versionString = versionString.substring(0, plusIndex);
    }
    response.put(ENGINE_VERSION_PROPERTY, versionString);
    return response;
  }
}
