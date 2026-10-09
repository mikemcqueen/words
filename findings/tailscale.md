# Syncing final/.wf and nutrimatic/idx over Tailscale

The goal is a one-way sync of `final/.wf` (9.3G) and `../nutrimatic/idx` (8.9G) from this machine's WSL into juniper's WSL. It replaces the rsync to `/mnt/juniper-repos/...` in the `sync.sh` scripts.

## What Tailscale is

- It's a private VPN between your own machines, built on WireGuard. Each machine you sign in gets a fixed `100.x.y.z` address and a short name (e.g. `juniper`) that only your other machines can reach.
- Traffic goes directly between the machines and is encrypted end to end. Tailscale's servers only swap public keys and help the machines find each other. If a direct path fails, traffic goes through their relay servers, still encrypted.
- Nothing gets opened to the internet. There's no port forwarding or router setup, and it works the same on your home network or away from it.
- **Cost:** the free Personal plan covers this use (up to 3 users and 100 devices).

## The WSL part

Install Tailscale **inside the WSL distro** on each machine. `/etc/wsl.conf` already has `systemd=true`, so `tailscaled` runs as a normal service. The distro then joins your Tailscale network as its own device with its own `100.x` address, and WSL's NAT stops mattering. This is simpler than running Tailscale on Windows and passing traffic into WSL, which needs Windows' "mirrored" networking mode plus firewall rules.

**Catch:** WSL shuts down its VM soon after the last terminal closes, even with systemd running. The destination's WSL has to be running when you sync. Leaving a terminal open is enough, or you can add a Windows logon task that runs `wsl.exe`.

## Setup

**Both machines (inside WSL):**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up        # prints a login URL; sign in with the same account on both
```
In the admin console (login.tailscale.com), turn off key expiry for the destination machine. Otherwise it drops off the network every 180 days until you sign in again.

**Destination only:**
```bash
sudo apt install openssh-server rsync
sudo systemctl enable --now ssh
# set PasswordAuthentication no in /etc/ssh/sshd_config, then restart ssh
```

**Source:** create a dedicated key with `ssh-keygen -t ed25519 -f ~/.ssh/sync_juniper` and add the `.pub` file to the destination's `~/.ssh/authorized_keys`.

**New `final/sync.sh`:**
```bash
#!/bin/bash
rsync -avhP -e "ssh -i ~/.ssh/sync_juniper" "$@" \
  ~/code/words/final/.wf/ mike@juniper:code/words/final/.wf/
```
Do the same for `idx`, and adjust the destination paths to wherever they live in juniper's WSL.

**Why this is faster than the mount:** with `/mnt/juniper-repos`, rsync sees both sides as local disk. It reads every remote file over the network and copies whole files. Over ssh, a second rsync runs on juniper and checks its own files on its own disk, so only the changed blocks cross the network. That's a big difference at this size, especially if the big files change only partly.

## Security

- **Network exposure:** sshd can only be reached through Tailscale, since WSL's NAT already blocks direct connections from your local network. The only machines that can connect are the ones signed into your Tailscale account.
- **Your Tailscale login is now the main thing to protect.** It's whatever account you sign in with (Google, GitHub, etc.), so turn on 2FA there. If you want, Tailscale's access rules can limit the source machine to port 22 on juniper and nothing else.
- **Limiting what a stolen key can do:** start that key's line in juniper's `authorized_keys` like this:
  ```
  command="rrsync /home/mike/code",restrict ssh-ed25519 AAAA... sync-key
  ```
  `rrsync` comes with rsync (on older distros it's a gzipped script under `/usr/share/doc/rsync/scripts`). That key can then only run rsync inside `~/code`: no shell, no port forwarding. Destination paths in the script become relative to that directory, e.g. `juniper:words/final/.wf/`.
- **Simpler option:** `sudo tailscale up --ssh` on juniper turns on Tailscale's built-in SSH instead of openssh. You don't manage any keys, and access rests entirely on your Tailscale account. That's simpler, but you lose the rrsync restriction.
