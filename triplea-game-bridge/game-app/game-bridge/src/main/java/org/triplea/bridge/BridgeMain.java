package org.triplea.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import games.strategy.triplea.settings.ClientSetting;
import io.javalin.Javalin;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.ThreadRunner;

@Slf4j
public final class BridgeMain {

  private static final int HTTP_PORT = 8081;
  private static final Gson GSON = new Gson();

  public static void main(final String[] args) {
    // Initialize client settings framework so that any TripleA code that reads
    // ClientSetting values (e.g. during GameObjectInputStream deserialization)
    // has a backing Preferences instance and does not throw IllegalStateException.
    ClientSetting.initialize();

    final String host = parseArg(args, "--host", "127.0.0.1");
    final int port = Integer.parseInt(parseArg(args, "--port", "3300"));
    final String name = parseArg(args, "--name", "Bot_Bridge");
    final String take = parseArg(args, "--take", "Japan");

    log.info("Bridge starting: host={}, port={}, name={}, take={}", host, port, name, take);

    final BridgeRuntime runtime = new BridgeRuntime(host, port, name, take);

    ThreadRunner.runInNewThread(
        () -> {
          try {
            runtime.connect();
            log.info("Bridge connected and took player: {}", take);
          } catch (Exception e) {
            log.error("Bridge connection failed", e);
          }
        });

    final Javalin app = Javalin.create();

    app.get("/health", ctx -> ctx.result("ok"));

    app.get(
        "/state",
        ctx -> {
          try {
            final Map<String, Object> state = runtime.getState();
            ctx.contentType("application/json");
            ctx.result(GSON.toJson(state));
          } catch (Exception e) {
            log.error("GET /state failed", e);
            ctx.status(500).contentType("application/json")
                .result(GSON.toJson(Map.of("error", e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName())));
          }
        });

    app.get(
        "/legal_actions",
        ctx -> {
          final List<Map<String, String>> actions = runtime.getLegalActions();
          ctx.contentType("application/json");
          ctx.result(GSON.toJson(actions));
        });

    app.post(
        "/act",
        ctx -> {
          final String body = ctx.body();
          final JsonObject action =
              body == null || body.isBlank()
                  ? new JsonObject()
                  : GSON.fromJson(body, JsonObject.class);
          final Map<String, Object> result = runtime.applyAction(action);
          ctx.contentType("application/json");
          ctx.result(GSON.toJson(result));
        });

    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  runtime.shutdown();
                  log.info("Bridge shutdown");
                }));

    app.start(HTTP_PORT);
    log.info("Bridge HTTP server listening on port {}", HTTP_PORT);
  }

  private static String parseArg(final String[] args, final String key, final String defaultValue) {
    for (int i = 0; i < args.length - 1; i++) {
      if (key.equals(args[i])) {
        return args[i + 1];
      }
    }
    return defaultValue;
  }

  private BridgeMain() {}
}
