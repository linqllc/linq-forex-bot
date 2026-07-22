# Security

Do not commit:

- broker API tokens or account IDs
- `.env` files
- private SSH keys
- production configuration containing secrets
- personally identifying account exports

Use environment variables or a local secret manager. Live trading must remain disabled unless paper-trading, reconciliation, risk-limit, and kill-switch validation has passed.
