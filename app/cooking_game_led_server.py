import json
import socket
import time

from machine import Pin
import neopixel


class CookingGameHandler:
    WIDTH = 16
    HEIGHT = 16
    NUM_LEDS = WIDTH * HEIGHT
    LED_PIN = 15
    MAX_WIFI_RETRIES = 20
    MAX_HEADER_SIZE = 4096
    MAX_BODY_SIZE = 65536

    # Most 16x16 NeoPixel matrices are wired serpentine.
    SERPENTINE = True
    X_REVERSED = False
    Y_REVERSED = False

    def __init__(self):
        self.np = neopixel.NeoPixel(Pin(self.LED_PIN), self.NUM_LEDS)
        self.clear_leds()

    def _normalize_xy(self, x, y):
        if self.X_REVERSED:
            x = self.WIDTH - 1 - x
        if self.Y_REVERSED:
            y = self.HEIGHT - 1 - y
        return x, y

    def xy_to_index(self, x, y):
        x, y = self._normalize_xy(x, y)

        if self.SERPENTINE and (y % 2 == 1):
            x = self.WIDTH - 1 - x

        return y * self.WIDTH + x

    def clear_leds(self):
        for i in range(self.NUM_LEDS):
            self.np[i] = (0, 0, 0)
        self.np.write()

    def clamp_color(self, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0

        if value < 0:
            return 0
        if value > 255:
            return 255
        return value

    def set_pixels(self, pixels):
        self.clear_leds()

        if not isinstance(pixels, list):
            pixels = []

        for p in pixels:
            if not isinstance(p, dict):
                continue

            try:
                x = int(p.get("x", 0))
                y = int(p.get("y", 0))
            except (TypeError, ValueError):
                continue

            if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
                idx = self.xy_to_index(x, y)
                r = self.clamp_color(p.get("r", 0))
                g = self.clamp_color(p.get("g", 0))
                b = self.clamp_color(p.get("b", 0))
                self.np[idx] = (r, g, b)

        self.np.write()

    def get_html(self):
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cooking with Ethan LED Matrix</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #10131a;
      color: #f3f4f6;
      margin: 0;
      padding: 20px;
    }
    .wrap {
      max-width: 760px;
      margin: 0 auto;
    }
    .panel {
      background: #1b2230;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
      margin-bottom: 16px;
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 12px 14px;
      background: #2563eb;
      color: white;
      font-size: 15px;
      font-weight: bold;
    }
    button.stop {
      background: #dc2626;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(16, 16px);
      gap: 3px;
      justify-content: center;
      margin-top: 18px;
    }
    .pixel {
      width: 16px;
      height: 16px;
      border-radius: 4px;
      background: #05070c;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hint {
      color: #cbd5e1;
      font-size: 14px;
      line-height: 1.5;
    }
    #status {
      margin-top: 8px;
      min-height: 20px;
      color: #93c5fd;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Cooking with Ethan LED Matrix</h1>
      <p class="hint">These buttons send animation frames to <code>/api/led</code>. If the matrix lights in the wrong order, change the mapping flags in the Python file.</p>
      <div class="buttons">
        <button onclick="runAnimation(testPattern)">Test Pattern</button>
        <button onclick="runAnimation(rainbowSweep)">Rainbow Sweep</button>
        <button onclick="runAnimation(pulse)">Pulse</button>
        <button onclick="runAnimation(snake)">Snake</button>
        <button class="stop" onclick="stopAnimation()">Stop / Clear</button>
      </div>
      <div id="status">Ready.</div>
      <div id="preview" class="grid"></div>
    </div>
  </div>

  <script>
    const WIDTH = 16;
    const HEIGHT = 16;
    const PULSE_BASE = 20;
    const PULSE_RANGE = 180;
    const PULSE_SPEED_DIVISOR = 2;
    const SNAKE_LENGTH = 18;
    const SNAKE_MIN_FADE = 30;
    const SNAKE_FADE_STEP = 14;
    const preview = document.getElementById("preview");
    const statusEl = document.getElementById("status");
    let timer = null;

    for (let i = 0; i < WIDTH * HEIGHT; i++) {
      const cell = document.createElement("div");
      cell.className = "pixel";
      preview.appendChild(cell);
    }

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function stopAnimation() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      sendPixels([]);
      drawPreview([]);
      setStatus("Stopped.");
    }

    async function sendPixels(pixels) {
      try {
        const response = await fetch("/api/led", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels })
        });

        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
      } catch (error) {
        setStatus("Send failed: " + error.message);
      }
    }

    function drawPreview(pixels) {
      const cells = preview.children;
      for (let i = 0; i < cells.length; i++) {
        cells[i].style.background = "#05070c";
      }
      for (const pixel of pixels) {
        const index = pixel.y * WIDTH + pixel.x;
        if (cells[index]) {
          cells[index].style.background = "rgb(" + pixel.r + "," + pixel.g + "," + pixel.b + ")";
        }
      }
    }

    function runAnimation(frameBuilder, interval = 120) {
      stopAnimation();
      let frame = 0;
      timer = setInterval(() => {
        const pixels = frameBuilder(frame++);
        drawPreview(pixels);
        sendPixels(pixels);
      }, interval);
      setStatus("Animation running.");
    }

    function hsvToRgb(h, s, v) {
      let r = 0, g = 0, b = 0;
      const i = Math.floor(h * 6);
      const f = h * 6 - i;
      const p = v * (1 - s);
      const q = v * (1 - f * s);
      const t = v * (1 - (1 - f) * s);
      switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
      }
      return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
      };
    }

    function testPattern() {
      const pixels = [];
      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          const left = x < WIDTH / 3;
          const middle = x >= WIDTH / 3 && x < (2 * WIDTH) / 3;
          pixels.push({
            x,
            y,
            r: left ? 255 : 0,
            g: middle ? 255 : 0,
            b: !left && !middle ? 255 : 0
          });
        }
      }
      return pixels;
    }

    function rainbowSweep(frame) {
      const pixels = [];
      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          const hue = ((x + y + frame) % (WIDTH + HEIGHT)) / (WIDTH + HEIGHT);
          const rgb = hsvToRgb(hue, 1, 0.35);
          pixels.push({ x, y, ...rgb });
        }
      }
      return pixels;
    }

    function pulse(frame) {
      const pixels = [];
      const value = Math.round((Math.sin(frame / PULSE_SPEED_DIVISOR) * 0.5 + 0.5) * PULSE_RANGE) + PULSE_BASE;
      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          pixels.push({ x, y, r: value, g: Math.round(value * 0.4), b: 0 });
        }
      }
      return pixels;
    }

    function snake(frame) {
      const pixels = [];
      const head = frame % (WIDTH * HEIGHT);
      for (let i = 0; i < SNAKE_LENGTH; i++) {
        const index = (head - i + WIDTH * HEIGHT) % (WIDTH * HEIGHT);
        const x = index % WIDTH;
        const y = Math.floor(index / WIDTH);
        const fade = Math.max(SNAKE_MIN_FADE, 255 - i * SNAKE_FADE_STEP);
        pixels.push({ x, y, r: 0, g: fade, b: Math.round(fade * 0.25) });
      }
      return pixels;
    }
  </script>
</body>
</html>
"""

    def connect_wifi(self, ssid, password):
        try:
            import network

            wlan = network.WLAN(network.STA_IF)
            wlan.active(True)
            wlan.connect(ssid, password)

            print("Connecting to WiFi...")
            for attempt in range(self.MAX_WIFI_RETRIES):
                if wlan.isconnected():
                    ip = wlan.ifconfig()[0]
                    print("Connected! IP:", ip)
                    return ip

                print("Waiting for WiFi... ({}/{})".format(attempt + 1, self.MAX_WIFI_RETRIES))
                time.sleep(1)

            print("WiFi connection failed.")
            return None
        except Exception as e:
            print("WiFi error:", e)
            return None

    def send_response(self, client_socket, body, content_type="text/html; charset=utf-8", status="200 OK"):
        if isinstance(body, str):
            body = body.encode("utf-8")

        response = b"HTTP/1.1 " + status.encode() + b"\r\n"
        response += b"Content-Type: " + content_type.encode() + b"\r\n"
        response += b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        response += b"Connection: close\r\n"
        response += b"Access-Control-Allow-Origin: *\r\n"
        response += b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        response += b"Access-Control-Allow-Headers: Content-Type\r\n"
        response += b"\r\n"
        response += body
        client_socket.sendall(response)

    def read_full_request(self, client_socket, initial_data):
        """Read a complete HTTP request and enforce header/body size limits."""
        data = initial_data

        while b"\r\n\r\n" not in data:
            if len(data) >= self.MAX_HEADER_SIZE:
                raise ValueError("request headers too large")
            chunk = client_socket.recv(512)
            if not chunk:
                return data
            data += chunk

        headers_end = data.find(b"\r\n\r\n")
        headers = data[:headers_end].decode("utf-8", "ignore")
        content_length = 0

        for line in headers.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except (TypeError, ValueError):
                    content_length = 0

        if content_length > self.MAX_BODY_SIZE:
            raise ValueError("request body too large")

        body_start = headers_end + 4
        header_bytes = data[:body_start]
        body = data[body_start:]

        while len(body) < content_length:
            chunk = client_socket.recv(1024)
            if not chunk:
                break
            body += chunk

        return header_bytes + body

    def handle_led_post(self, client_socket, request_data):
        try:
            request_text = request_data.decode("utf-8", "ignore")
            parts = request_text.split("\r\n\r\n", 1)
            if len(parts) < 2:
                self.send_response(
                    client_socket,
                    '{"status":"error","message":"no body"}',
                    "application/json",
                    "400 Bad Request",
                )
                return

            body = parts[1]
            data = json.loads(body)

            if isinstance(data, list):
                pixels = data
            else:
                pixels = data.get("pixels", [])

            self.set_pixels(pixels)
            self.send_response(client_socket, '{"status":"ok"}', "application/json")
        except Exception as e:
            print("LED POST error:", e)
            self.send_response(client_socket, '{"status":"error"}', "application/json", "500 Internal Server Error")

    def handle_request(self, client_socket, request_data):
        try:
            request_data = self.read_full_request(client_socket, request_data)
            request_text = request_data.decode("utf-8", "ignore")
            request_line = request_text.split("\r\n")[0]
            print("Request:", request_line)

            parts = request_line.split(" ")
            if len(parts) < 2:
                self.send_response(client_socket, "Bad Request", "text/plain", "400 Bad Request")
                return

            method = parts[0]
            path = parts[1].split("?", 1)[0]

            if method == "OPTIONS":
                self.send_response(client_socket, b"", "text/plain", "204 No Content")
            elif method == "GET" and (path == "/" or path == "/index.html"):
                self.send_response(client_socket, self.get_html())
            elif method == "POST" and path == "/api/led":
                self.handle_led_post(client_socket, request_data)
            elif method == "GET" and path == "/api/health":
                self.send_response(client_socket, '{"status":"ok"}', "application/json")
            else:
                self.send_response(client_socket, "<h1>404 - Page Not Found</h1>", "text/html; charset=utf-8", "404 Not Found")

        except Exception as e:
            print("Request error:", e)
            try:
                self.send_response(client_socket, "Internal Server Error", "text/plain", "500 Internal Server Error")
            except Exception:
                pass
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    def run(self, ssid, password, port=8000):
        device_ip = self.connect_wifi(ssid, password)
        if device_ip is None:
            print("Cannot start server without WiFi.")
            return

        addr = socket.getaddrinfo("0.0.0.0", port)[0][-1]
        server_socket = socket.socket()
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(addr)
        server_socket.listen(1)

        print("")
        print("========================================")
        print(" Cooking with Ethan server started")
        print(" Open in browser: http://{}:{}/".format(device_ip, port))
        print(" LED sync endpoint: http://{}:{}/api/led".format(device_ip, port))
        print("========================================")
        print("")

        try:
            while True:
                client_socket, client_addr = server_socket.accept()
                print("Client:", client_addr)
                request_data = client_socket.recv(1024)
                if request_data:
                    self.handle_request(client_socket, request_data)
        except KeyboardInterrupt:
            print("Server stopped.")
        finally:
            self.clear_leds()
            try:
                server_socket.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        from cooking_game_led_config import WIFI_SSID, WIFI_PASSWORD
    except ImportError:
        WIFI_SSID = None
        WIFI_PASSWORD = None

    if not WIFI_SSID or not WIFI_PASSWORD:
        print("Create app/cooking_game_led_config.py with WIFI_SSID and WIFI_PASSWORD before running.")
    else:
        game = CookingGameHandler()
        game.run(WIFI_SSID, WIFI_PASSWORD, 8000)
