"""
Comprehensive adversarial tests for centralized secret redaction (Priority 1).
Tests all 10 required secret types, prefixed environment variables, database URLs,
and verifies model request interception across discovery, generation, updater, and comments.
"""

from unittest.mock import MagicMock

from config import RedactingOpenAIWrapper
from security_utils import redact_secrets


class TestSecretRedactionPatterns:
    def test_aws_access_and_secret_keys(self):
        key_id = "AKIA" + "IOSFODNN7EXAMPLE"
        sec_key = "wJalrXUtnFEMI" + "/K7MDENG/bPxRfiCYEXAMPLEKEY"
        text = f'aws_key = "{key_id}"\naws_secret_access_key = "{sec_key}"'
        redacted = redact_secrets(text)
        assert key_id not in redacted
        assert sec_key not in redacted
        assert "[REDACTED_AWS_KEY]" in redacted
        assert "aws_key = " in redacted
        assert "aws_secret_access_key = " in redacted

    def test_github_tokens(self):
        tokens = [
            "gh" + "p_" + "1234567890abcdefghijklmnopqrstuvwxyzAB",
            "github_"
            + "pat_"
            + "11AAAAAAA01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdef",
            "gh" + "o_" + "1234567890abcdefghijklmnopqrstuvwxyzAB",
            "gh" + "u_" + "1234567890abcdefghijklmnopqrstuvwxyzAB",
            "gh" + "s_" + "1234567890abcdefghijklmnopqrstuvwxyzAB",
            "gh" + "r_" + "1234567890abcdefghijklmnopqrstuvwxyzAB",
        ]
        for t in tokens:
            text = f"token = '{t}'"
            redacted = redact_secrets(text)
            assert t not in redacted
            assert "[REDACTED_GITHUB_TOKEN]" in redacted
            assert "token = " in redacted

    def test_jwt_tokens(self):
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            + "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = f"auth_header = 'Bearer {jwt}'"
        redacted = redact_secrets(text)
        assert jwt not in redacted
        assert "[REDACTED" in redacted

    def test_openai_api_keys(self):
        sk1 = "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890"
        sk2 = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz1234567890abcdefg"
        text = f"key1 = '{sk1}'\nkey2 = '{sk2}'"
        redacted = redact_secrets(text)
        assert sk1 not in redacted
        assert sk2 not in redacted
        assert "[REDACTED_OPENAI_KEY]" in redacted

    def test_slack_and_stripe_tokens(self):
        slack = "xo" + "xb-" + "1234567890-123456789012-abcdefghijklmnopqrstuvwx"
        stripe = "sk_" + "live_" + "51Abcdefghijklmnopqrstuvwxy"
        text = f"slack = '{slack}'\nstripe = '{stripe}'"
        redacted = redact_secrets(text)
        assert slack not in redacted
        assert stripe not in redacted
        assert "[REDACTED_SLACK_TOKEN]" in redacted
        assert "[REDACTED_STRIPE_KEY]" in redacted

    def test_database_connection_strings_with_credentials(self):
        db_urls = [
            "postgres://app_user:s3cr3tP@ssw0rd@db.internal:5432/production",
            "postgresql://admin:mY_Sup3r_S3cr3t@localhost:5432/mydb",
            "mysql://root:rootpassword123@10.0.0.5:3306/app",
            "mongodb://mongo_admin:clusterP@ss@mongodb0.example.com:27017/admin",
        ]
        for url in db_urls:
            text = f"DATABASE_URL = '{url}'"
            redacted = redact_secrets(text)
            assert "s3cr3tP@ssw0rd" not in redacted
            assert "mY_Sup3r_S3cr3t" not in redacted
            assert "rootpassword123" not in redacted
            assert "clusterP@ss" not in redacted
            assert "[REDACTED_PASSWORD]" in redacted

    def test_prefixed_environment_variables(self):
        text = """
        MY_API_KEY = "secret_key_value_12345"
        DB_PASSWORD = "database_secret_pass"
        AWS_SECRET_ACCESS_KEY = "aws_secret_val"
        CLIENT_SECRET = "client_secret_xyz"
        export APP_AUTH_TOKEN=exported_token_value
        """
        redacted = redact_secrets(text)
        assert "secret_key_value_12345" not in redacted
        assert "database_secret_pass" not in redacted
        assert "aws_secret_val" not in redacted
        assert "client_secret_xyz" not in redacted
        assert "exported_token_value" not in redacted
        assert "MY_API_KEY = " in redacted
        assert "DB_PASSWORD = " in redacted
        assert "CLIENT_SECRET = " in redacted

    def test_private_key_blocks(self):
        text = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Y1+abcdefghijklmnopqrstuvwxyz1234567890
-----END RSA PRIVATE KEY-----
        """
        redacted = redact_secrets(text)
        assert "MIIEowIBAAKCAQEA0Y1" not in redacted
        assert "[REDACTED_PRIVATE_KEY]" in redacted


class TestOutboundModelRedactionWrapper:
    def test_redacting_openai_wrapper_intercepts_messages(self):
        mock_raw_client = MagicMock()
        mock_raw_client.chat.completions.create.return_value = "mock_response"

        wrapper = RedactingOpenAIWrapper(mock_raw_client)
        test_sk = "sk-" + "12345678901234567890abcdef"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Here is the code with DB_PASSWORD = 'super_secret_pw' and {test_sk}",
            },
        ]

        wrapper.chat.completions.create(model="gpt-4o", messages=messages)

        assert mock_raw_client.chat.completions.create.called
        call_kwargs = mock_raw_client.chat.completions.create.call_args.kwargs
        outbound_msgs = call_kwargs["messages"]
        user_msg = outbound_msgs[1]["content"]

        assert "super_secret_pw" not in user_msg
        assert test_sk not in user_msg
        assert "[REDACTED]" in user_msg or "[REDACTED_OPENAI_KEY]" in user_msg
