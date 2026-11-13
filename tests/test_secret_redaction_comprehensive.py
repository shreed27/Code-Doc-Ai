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

    def test_demonstrated_leaks_unquoted_and_quoted(self):
        # 1. Unquoted client_secret: value (YAML / prose)
        yaml_text = "auth:\n  client_id: test_id\n  client_secret: unquoted_secret_val_123"
        redacted_yaml = redact_secrets(yaml_text)
        assert "unquoted_secret_val_123" not in redacted_yaml
        assert "client_secret: [REDACTED]" in redacted_yaml
        assert "client_id: test_id" in redacted_yaml

        # 2. Shell DB_PASSWORD=value
        env_text = "DB_USER=postgres\nDB_PASSWORD=my_db_secret_pass_456\nPORT=5432"
        redacted_env = redact_secrets(env_text)
        assert "my_db_secret_pass_456" not in redacted_env
        assert "DB_PASSWORD=[REDACTED]" in redacted_env
        assert "PORT=5432" in redacted_env

        # 3. password: value
        pw_text = "credentials:\n  username: admin\n  password: secret_admin_pass_789"
        redacted_pw = redact_secrets(pw_text)
        assert "secret_admin_pass_789" not in redacted_pw
        assert "password: [REDACTED]" in redacted_pw

        # 4. api_key = value (Python / config)
        py_text = "api_key = my_api_key_value_999\nservice_url = 'https://api.test'"
        redacted_py = redact_secrets(py_text)
        assert "my_api_key_value_999" not in redacted_py
        assert "api_key = [REDACTED]" in redacted_py

        # 5. X-API-Key: value (HTTP header)
        header_text = "GET /v1/data HTTP/1.1\nX-API-Key: x_header_secret_abc123\nHost: api.test"
        redacted_header = redact_secrets(header_text)
        assert "x_header_secret_abc123" not in redacted_header
        assert "X-API-Key: [REDACTED]" in redacted_header

        # 6. Dictionary-style "Authorization": "Basic ..."
        dict_text = (
            '{"Authorization": "Basic dXNlcjpwYXNzMTIzNDU2", "Content-Type": "application/json"}'
        )
        redacted_dict = redact_secrets(dict_text)
        assert "dXNlcjpwYXNzMTIzNDU2" not in redacted_dict
        assert "[REDACTED_TOKEN]" in redacted_dict

        # 7. Normal prose must NOT be blindly redacted
        prose_text = "Please enter your password to log in. The API key must be kept secret."
        redacted_prose = redact_secrets(prose_text)
        assert "enter your password to log in" in redacted_prose
        assert "The API key must be kept secret" in redacted_prose

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

    def test_serialized_http_payload_never_leaks_secrets(self):
        import json

        mock_raw_client = MagicMock()
        planted_secrets = [
            "planted_db_pass_12345",
            "planted_client_secret_67890",
            "planted_api_key_abcde",
            "planted_password_secret_pass",
            "planted_x_api_key_header_val",
            "dXNlcjpwYXNzMTIzNDU2",
        ]

        def fake_create(*args, **kwargs):
            # Capture actual serialized payload as it would travel over the wire
            serialized = json.dumps(kwargs)
            for secret in planted_secrets:
                assert secret not in serialized
            return {"choices": [{"message": {"content": "ok"}}]}

        mock_raw_client.chat.completions.create.side_effect = fake_create
        wrapper = RedactingOpenAIWrapper(mock_raw_client)

        payload_content = (
            f"Config:\n"
            f"DB_PASSWORD={planted_secrets[0]}\n"
            f"client_secret: {planted_secrets[1]}\n"
            f"api_key = {planted_secrets[2]}\n"
            f"credentials:\n  password: {planted_secrets[3]}\n"
            f"Headers:\nX-API-Key: {planted_secrets[4]}\n"
            f'{{"Authorization": "Basic {planted_secrets[5]}"}}\n'
        )
        wrapper.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": payload_content}],
        )
