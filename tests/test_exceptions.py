"""Unit tests for exceptions."""

import pytest
from llm_manager.exceptions import (
    LLMProviderError,
    UnknownProviderError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    TokenLimitError,
    InvalidRequestError,
    ProviderUnavailableError,
)


class TestExceptions:
    """Tests for LLM Manager exceptions."""

    def test_base_exception(self):
        """Test base LLMProviderError."""
        with pytest.raises(LLMProviderError):
            raise LLMProviderError("Test error")

    def test_unknown_provider_error(self):
        """Test UnknownProviderError is subclass of LLMProviderError."""
        assert issubclass(UnknownProviderError, LLMProviderError)
        with pytest.raises(UnknownProviderError):
            raise UnknownProviderError("Unknown provider")

    def test_api_connection_error(self):
        """Test APIConnectionError."""
        assert issubclass(APIConnectionError, LLMProviderError)
        with pytest.raises(APIConnectionError):
            raise APIConnectionError("Connection failed")

    def test_authentication_error(self):
        """Test AuthenticationError."""
        assert issubclass(AuthenticationError, LLMProviderError)
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Invalid API key")

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        assert issubclass(RateLimitError, LLMProviderError)
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded")

    def test_token_limit_error(self):
        """Test TokenLimitError."""
        assert issubclass(TokenLimitError, LLMProviderError)
        with pytest.raises(TokenLimitError):
            raise TokenLimitError("Token limit exceeded")

    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        assert issubclass(InvalidRequestError, LLMProviderError)
        with pytest.raises(InvalidRequestError):
            raise InvalidRequestError("Invalid request")

    def test_provider_unavailable_error(self):
        """Test ProviderUnavailableError."""
        assert issubclass(ProviderUnavailableError, LLMProviderError)
        with pytest.raises(ProviderUnavailableError):
            raise ProviderUnavailableError("Provider unavailable")
