# Security policy

This is a proof of concept that presents a synthetic device identity on a
network. Do not run it on hardware or networks you do not control.

Report vulnerabilities privately to the maintainer instead of opening a public
issue. Include the affected version or commit, the smallest reproduction you
can, and whether the issue also affects a real AI Key.

The adopt token and TLS private key are written mode 0600 and must stay out of
version control. The emulator accepts the console's adoption POST without
authentication, matching the real device, so never expose port 8080 to an
untrusted network.
