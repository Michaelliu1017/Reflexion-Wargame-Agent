package games.strategy.engine.data;

import java.io.Serializable;
import java.util.Optional;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * Serializable stand-in for Optional&lt;String&gt; so that RPC return values (e.g. from
 * IAbstractPlaceDelegate.placeUnits) can be sent over the network without NotSerializableException.
 */
@NoArgsConstructor
@AllArgsConstructor
@Getter
public class OptionalStringHolder implements Serializable {
  private static final long serialVersionUID = 1L;
  private boolean empty;
  private String value;

  public static OptionalStringHolder of(Optional<String> opt) {
    if (opt == null || opt.isEmpty()) {
      return new OptionalStringHolder(true, null);
    }
    return new OptionalStringHolder(false, opt.get());
  }

  public Optional<String> toOptional() {
    return empty ? Optional.empty() : Optional.ofNullable(value);
  }
}
