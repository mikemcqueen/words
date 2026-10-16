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
```

**Source:** create a dedicated key and add it to juniper. Do this before turning off password login, since `ssh-copy-id` logs in with your password once:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/sync_juniper
ssh-copy-id -i ~/.ssh/sync_juniper.pub mike@juniper
ssh -i ~/.ssh/sync_juniper mike@juniper true     # returns with no password prompt if the key works
```
`~/.ssh/authorized_keys` is a plain text file with one public key per line. `ssh-copy-id` appends your key to it, creating the file if needed and setting its permissions.

If password login is already off, add the key by hand. Run `cat ~/.ssh/sync_juniper.pub` on the source and copy the single line (`ssh-ed25519 AAAA... mike@host`), then on juniper:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA... mike@host' >> ~/.ssh/authorized_keys   # >> appends; > would wipe existing keys
chmod 600 ~/.ssh/authorized_keys
```
sshd quietly ignores `authorized_keys` if it, `~/.ssh` or your home directory can be written by anyone but you. If key login fails, check those permissions first (`chmod go-w ~` fixes a group-writable home).

**Destination, once the key works:** set `PasswordAuthentication no` in `/etc/ssh/sshd_config`, then restart and check (the service is `ssh`, not `sshd`):
```bash
sudo sshd -t                                     # syntax check; no output means OK
sudo systemctl restart ssh
sudo sshd -T | grep -i passwordauthentication    # should print: passwordauthentication no
```
If it still says `yes`, a file in `/etc/ssh/sshd_config.d/` is overriding it. Those files are included at the top of `sshd_config`, and sshd uses the first value it finds for each setting. Run `grep -ri passwordauthentication /etc/ssh/sshd_config.d/`, then change that line to `no` or delete it, and restart again.
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
- **Limiting what a stolen key can do:** edit that key's line in juniper's `authorized_keys` so it starts like this, with the restriction in front of `ssh-ed25519`:
  ```
  command="rrsync /home/mike/code",restrict ssh-ed25519 AAAA... sync-key
  ```
  `rrsync` comes with rsync (on older distros it's a gzipped script under `/usr/share/doc/rsync/scripts`). That key can then only run rsync inside `~/code`: no shell, no port forwarding. Destination paths in the script become relative to that directory, e.g. `juniper:words/final/.wf/`.
- **Simpler option:** `sudo tailscale up --ssh` on juniper turns on Tailscale's built-in SSH instead of openssh. You don't manage any keys, and access rests entirely on your Tailscale account. That's simpler, but you lose the rrsync restriction.
