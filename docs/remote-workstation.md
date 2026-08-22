# Remote workstation setup

Turning a machine at home into something you can fully use from anywhere, for
anything — not just for this project.

Assumes the GPU box runs Windows with WSL2 (Ubuntu) and you drive it from a
laptop. Adjust the Windows-specific parts if it runs Linux natively.

---

## The three foundations

| layer | tool | what it gives you |
|---|---|---|
| network | **Tailscale** | a private network between your devices, no ports open to the internet, works behind any NAT |
| shell | **OpenSSH** | terminal access, file transfer, port forwarding |
| desktop | **RustDesk** | the full graphical desktop when you need GUI apps |

Everything below builds on those.

---

## 1. Sessions that survive disconnection

Without this, closing your laptop kills whatever is running. This is the single
most important addition after SSH.

```bash
sudo apt install -y tmux
```

Daily use:

```bash
tmux new -s main         # start a named session
# work normally; press Ctrl-b then d to detach
tmux attach -t main      # reattach later, from any device
tmux ls                  # list sessions
```

Detach, close the laptop, reconnect from your phone three hours later, and the
session is exactly where you left it. Anything long-running should start inside
tmux.

A useful default — never think about it again:

```bash
echo '[ -z "$TMUX" ] && [ -n "$SSH_CONNECTION" ] && tmux attach -t main || tmux new -s main' >> ~/.bashrc
```

---

## 2. Editing code with a real editor

**VS Code Remote-SSH** — the editor runs on your laptop, everything executes on
the remote machine.

1. Install VS Code, add the **Remote - SSH** extension
2. `Cmd/Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → `user@hostname`
3. File → Open Folder → whatever you are working on

The integrated terminal, debugger, and file tree all operate remotely. Port
forwarding is automatic: start a web server on the remote machine and VS Code
offers it on your local `localhost`.

**From a device with no VS Code** (a phone, an iPad, a borrowed laptop), use a
browser-based tunnel instead:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
./code tunnel
```

It prints a `vscode.dev` URL that works in any browser. Run it inside tmux so it
stays up.

---

## 3. Moving files

Three options, in order of how often you will reach for them:

```bash
# Pull results back, resumable, only transfers what changed
rsync -avP user@host:~/project/runs/ ./runs/

# Push a single file
scp report.pdf user@host:~/

# Taildrop: no SSH involved, works to phones too
tailscale file cp slides.pdf hostname:
tailscale file get ~/Downloads      # on the receiving device
```

For occasional browsing, VS Code's file explorer is usually easier than mounting
a network drive.

---

## 4. Access from a phone

Genuinely useful for checking on a long job.

1. **Tailscale** app (iOS/Android) — sign in, same account
2. **Termius** or **Blink Shell** — SSH client; add the host and your key
3. **RustDesk** mobile — the full desktop if you need it

With tmux, you can attach to the same session you left on your laptop.

---

## 5. Getting notified when something finishes

Long jobs should tell you they are done rather than you checking.

```bash
# Pick any unguessable topic name; no account needed
curl -d "training finished on gpubox" ntfy.sh/pick-something-unguessable-here
```

Install the **ntfy** app on your phone and subscribe to that topic. Then append
it to anything slow:

```bash
python train.py; curl -d "train.py done (exit $?)" ntfy.sh/your-topic
```

Note the topic is effectively a password — anyone who guesses it can read your
notifications, so make it long and random.

---

## 6. Monitoring the machine

```bash
sudo apt install -y btop      # CPU, memory, processes
nvidia-smi                     # GPU: works from inside WSL
nvidia-smi dmon -s um          # continuous GPU utilisation
pipx install nvitop && nvitop  # nicer combined view
```

To watch a log from another machine without installing anything:

```bash
ssh user@host 'tail -f ~/project/runs/latest/metrics.jsonl'
```

---

## 7. Using your home connection from anywhere

Tailscale can route your laptop's traffic through the home machine — useful on
public Wi-Fi, or to appear on your home network.

On the GPU box:

```powershell
tailscale set --advertise-exit-node
```

Approve it in the Tailscale admin console, then select it as an exit node from
the Tailscale menu on your laptop.

---

## 8. Power

The machine must stay awake to be reachable. Idle draw is roughly 10–15 W,
around £2–3/month in the UK.

Keep it awake (PowerShell as Administrator):

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

If you want it genuinely off when unused, the only reliable remote power switch
is a **smart plug** plus a BIOS setting:

1. BIOS/UEFI → "Restore on AC Power Loss" → **Power On**
2. Shut down cleanly in software, then cut the plug from your phone
3. Flip the plug back on to boot it

Wake-on-LAN is the alternative but needs another always-on device on the same
network to send the magic packet, which usually defeats the point.

---

## 9. Security

The realistic threat is someone sitting down at the machine, not a remote
attacker — Tailscale means nothing is exposed to the internet.

- **`Win+L` before you walk away.** SSH sessions and running jobs continue
  underneath; anyone at the keyboard sees a lock screen.
- **RustDesk Privacy Mode** blanks the physical screen and blocks the local
  keyboard while you are connected.
- **Device encryption / BitLocker** on. Without it, physical access is total
  access — someone boots a USB stick and reads the disk.
- **Key-based SSH only.** On Windows, an administrator account authenticates
  against `C:\ProgramData\ssh\administrators_authorized_keys`, *not*
  `~/.ssh/authorized_keys`, and that file's ACL must contain only
  `Administrators` and `SYSTEM` or sshd ignores it silently.
- **Tailscale admin console** → enable device approval, and review connected
  devices occasionally.

---

## Daily workflow, once this is set up

```bash
ssh user@host            # or click the host in VS Code
tmux attach -t main      # back exactly where you were
```

Start long jobs inside tmux with an ntfy call on the end. Close the laptop
whenever. Reconnect from anything.
