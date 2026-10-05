# Speedometer 3.1 (Chrome)

Runs the [Speedometer 3.1](https://browserbench.org/Speedometer3.1/) browser benchmark in Google Chrome on macOS and prints the score.

## How it Works

*   `speedometer.mjs` closes any Chrome left open by a previous run, then launches a fresh Chrome window (empty profile in `./chrome-profile`, remote debugging on port 9222) through `open`, so Chrome is not tied to the script or an SSH session.
*   It attaches with `puppeteer-core` and loads `https://browserbench.org/Speedometer3.1/?startAutomatically=true`, which starts the benchmark without clicking "Start Test".
*   It waits for the results summary, then prints one line of JSON: score, ± confidence, user agent, and run time in seconds.
*   It detaches without closing Chrome, so the score stays on screen.

Chrome runs with its normal settings. Puppeteer's automation flags are not added.

## Requirements

*   macOS with Google Chrome in `/Applications`
*   Node.js 18+
*   A user logged in at the Mac's screen (Chrome opens a visible window)

## Usage

```bash
npm install
node speedometer.mjs
```

Output:

```
{"score":"37.9","confidence":"± 2.6","ua":"Mozilla/5.0 (Macintosh; ...) Chrome/154.0.0.0 Safari/537.36","seconds":25}
```

Several runs in a row:

```bash
for i in 1 2 3; do node speedometer.mjs; sleep 10; done
```

Over SSH, use the full path to `node`, since a remote command doesn't load the shell profile:

```bash
ssh <mac> 'cd ~/machine_tests/browser/speedometer && /opt/homebrew/bin/node speedometer.mjs'
```

## Tips

*   Leave the Mac idle during a run and check that nothing heavy is running in the background (`top`).
*   Plug in power for consistent results.

## Example Results

| Machine | Browser | Score |
|---|---|---|
| MacBook Air M1 (16GB), macOS 27.0.1 | Chrome 154 | 37.6, 37.6, 37.5, 37.9, 37.8 |
