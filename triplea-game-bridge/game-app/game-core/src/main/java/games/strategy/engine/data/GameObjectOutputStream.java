package games.strategy.engine.data;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Optional;

/**
 * To maintain == relationships and the singleton nature of many classes in GameData we do some work
 * in the ObjectSteam. For example, when we serialize a Territory over a GameObjectOutputStream, we
 * do not send an instance of Territory, but rather a marker saying that this is the territory, and
 * this is its name. When it comes time for a GameObjectOutputStream to read the territory on the
 * other side, the territory name is read, and the territory returned by the GameObjectInputStream
 * is the territory with that name belonging to the GameData associated with the
 * GameObjectInputStream. This ensures the state of the territory remains consistent.
 */
public class GameObjectOutputStream extends ObjectOutputStream {

  public GameObjectOutputStream(final OutputStream output) throws IOException {
    super(output);
    enableReplaceObject(true);
  }

  @Override
  protected Object replaceObject(final Object obj) {
    if (obj instanceof Named named && GameObjectStreamData.canSerialize(named)) {
      return new GameObjectStreamData(named);
    }
    if (obj instanceof Optional) {
      Optional<?> opt = (Optional<?>) obj;
      if (opt.isEmpty() || opt.get() instanceof String) {
        Optional<String> str =
            opt.isEmpty() ? Optional.empty() : Optional.of((String) opt.get());
        return OptionalStringHolder.of(str);
      }
    }

    return obj;
  }
}
