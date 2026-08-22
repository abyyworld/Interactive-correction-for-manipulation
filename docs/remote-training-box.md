# Running this project on a headless training box

The workflow this repo assumes: a GPU machine (RTX 4060) left at home running
Ubuntu headless, driven over SSH from a laptop. Training runs for hours, so it
has to survive dropped connections, closed lids and reboots — and the machine
has to be safe to leave physically unattended.

Nothing here is required to *use* the code. It is the setup the defaults were
chosen for.

---

## 1. Stop the laptop sleeping when you close the lid

The single most common "my server vanished" cause. A laptop-as-server suspends
on lid close by default and every SSH session dies with it.

```bash
sudo tee -a /etc/systemd/logind.conf >/dev/null <<'EOF'
HandleLidSwitch=ignore
HandleLidSwitchDocked=ignore
HandleLidSwitchExternalPower=ignore
EOF
sudo systemctl restart systemd-logind
```

Also stop idle suspend (safe on a machine you want reachable):

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Screen blanking is fine to leave on — it saves power and does not stop compute.

---

## 2. Remote access without exposing SSH to the internet

Port-forwarding SSH to the open internet means constant brute-force traffic and
one CVE away from a bad day. Use a private overlay network instead. Tailscale is
free for personal use and traverses university/home NAT without port forwarding.

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
```

On the laptop, install Tailscale and log into the same account. The box is then
reachable as `ssh you@gpubox` from anywhere, with no open ports.

Lock the firewall down to match:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on tailscale0
sudo ufw enable
```

If you prefer plain SSH over Tailscale rather than Tailscale SSH, harden it:

```bash
sudo tee /etc/ssh/sshd_config.d/99-hardening.conf >/dev/null <<'EOF'
PasswordAuthentication no
PermitRootLogin no
KbdInteractiveAuthentication no
EOF
sudo systemctl restart ssh
```

Copy your key up **before** disabling passwords, or you will lock yourself out:
`ssh-copy-id you@gpubox`.

---

## 3. Stopping other people using the machine

Threats in rough order of likelihood for a machine in a shared house or lab.

### Someone sits down at it while you are away

Disable auto-login and make it lock on idle:

```bash
# GNOME: lock after 5 minutes idle, no grace period
gsettings set org.gnome.desktop.session idle-delay 300
gsettings set org.gnome.desktop.screensaver lock-enabled true
gsettings set org.gnome.desktop.screensaver lock-delay 0
```

Auto-login off: `/etc/gdm3/custom.conf` must have `AutomaticLoginEnable=false`.

Make your home directory unreadable to other local accounts:

```bash
chmod 700 /home/$USER
```

Lock the physical console *right now* from your remote session:

```bash
loginctl lock-sessions
```

See who is logged in, and remove them:

```bash
who                          # local + remote sessions
loginctl list-sessions
sudo loginctl terminate-session <SESSION_ID>
sudo pkill -KILL -u <username>
```

Make sure nobody else holds sudo: `getent group sudo`. Remove with
`sudo deluser <username> sudo`.

If housemates genuinely need the machine, give them their own non-sudo account
rather than sharing yours — your files stay unreadable thanks to `chmod 700`.

### Someone reboots it to get in

This is the gap people miss. Without the two steps below, anyone at the keyboard
can edit the GRUB entry to `init=/bin/bash` and get a root shell **with no
password at all**, or boot a USB stick and read your disk.

**GRUB password** — blocks editing boot entries:

```bash
grub-mkpasswd-pbkdf2        # copy the grub.pbkdf2.sha512.... hash it prints
sudo tee /etc/grub.d/40_custom >/dev/null <<'EOF'
set superusers="admin"
password_pbkdf2 admin grub.pbkdf2.sha512.PASTE_YOUR_HASH_HERE
EOF
sudo update-grub
```

Add `--unrestricted` to the default menu entry if you want unattended reboots to
still boot without a prompt (recommended for a remote box — otherwise a power cut
leaves it sitting at a password prompt until you are physically there).

**UEFI/BIOS password + disable USB boot** — done in firmware setup at boot, not
from Linux. Set a supervisor password, disable boot from USB/network, and set the
internal SSD first in the boot order.

### Someone takes the disk out

Only **full-disk encryption** stops this. LUKS is an install-time choice in the
Ubuntu installer ("Encrypt the new Ubuntu installation"). Retrofitting means a
reinstall, so decide now if it matters to you.

Caveat for a remote box: an encrypted root asks for a passphrase at every boot,
before the network is up. After a power cut the machine will not come back until
you type it in physically. Options: accept that, or use `dropbear-initramfs` to
unlock over SSH, or encrypt only `/home` and leave root unencrypted.

Honest assessment for a typical shared house: GRUB password + UEFI password +
screen lock + `chmod 700` covers the realistic risk. FDE matters if the machine
could be stolen or if you hold data you are contractually obliged to protect.

---

## 4. Keeping long runs alive

Never start a training run in a bare SSH session — the run dies with the
connection.

```bash
tmux new -s train                    # start
# ... launch training ...
# detach with Ctrl-b then d
tmux attach -t train                 # reattach later, from any machine
```

Let your processes survive logout entirely:

```bash
sudo loginctl enable-linger $USER
```

Every training entry point in this repo checkpoints on a fixed interval and
resumes with `--resume`, so a reboot costs minutes rather than the whole run.

---

## 5. Watching a run from the other laptop

Metrics are written as JSON Lines, deliberately, so monitoring needs no extra
services:

```bash
ssh gpubox 'tail -f ~/Interactive-correction-for-manipulation/runs/<run_id>/metrics.jsonl'
```

GPU utilisation:

```bash
ssh gpubox 'nvidia-smi dmon -s um'
pipx install nvitop && ssh gpubox nvitop     # nicer, optional
```

Pull rendered evaluation GIFs back to look at:

```bash
rsync -avP gpubox:~/Interactive-correction-for-manipulation/runs/<run_id>/media/ ./media/
```
