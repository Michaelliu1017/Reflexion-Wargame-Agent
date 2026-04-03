# Headless Game Server

A headless game server for TripleA, also known as a _bot_.

The headless game server solves the problem of port forwarding to host games.

Future plan for bot servers is to replace with "network relay". Rather than
having an "automated host", the network relay would be transparent and would
only serve to simply relay network messages.

## Run

### Start a Lobby

Setup, make triplea directory and clone the needed code bases:
```
mkdir ~/work/triplea/
cd ~/work/triplea/
git clone ....triplea/lobby-server
git clone ....triplea/triplea-game
```

Start lobby on localhost, lobby will be running on localhost, port 3000:
```
cd work/triplea/lobby-server
# starts up the lobby
docker compose-up
```

### Running bot via Gradle

**MAPS_FOLDER must be a real directory that exists.** Do not use the placeholder `/path/to/downloadedMaps`.

1. Create a maps directory (if needed):
   ```bash
   mkdir -p $HOME/triplea/downloadedMaps
   ```
2. Run the headless server (macOS/Linux):
   ```bash
   MAPS_FOLDER=$HOME/triplea/downloadedMaps ./gradlew :game-app:game-headless:run
   ```
   On Linux with a different path:
   ```bash
   MAPS_FOLDER=/home/$USER/triplea/downloadedMaps ./gradlew :game-app:game-headless:run
   ```

You need a running Lobby for the bot to connect to (see "Start a Lobby" above). See `build.gradle` to change lobby URI, bot port, or bot name.

### Running bot via Docker

```
cd work/triplea/triplea/game-app/game-headless/
./clean-run-docker.sh
```

### Connect to local lobby from game app:

Start Triplea-Game, in 'settings' > 'testing', update Lobby URI to be: `http://localhost:3000`,
'save' & then connect to lobby via 'play online'.

### Running bots (on prod)

### Linux:

Below script opens firewall port 4000, downloads latest bot image & starts it.

```bash
BOT_NAME=....change_me.....

sudo ufw allow 4000
docker pull ghcr.io/triplea-game/bot:latest
MAPS_FOLDER=/home/$USER/triplea/downloadedMaps
docker run \
    --env BOT_NAME=$BOT_NAME \
    --env LOBBY_URI=https://prod.triplea-game.org \
    -v $MAPS_FOLDER:/downloadedMaps \
    -p 4000:4000 \
    ghcr.io/triplea-game/bot:latest
```
