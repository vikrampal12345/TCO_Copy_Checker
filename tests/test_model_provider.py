import json

from app.core.model_provider import (
    get_model_provider,
)


def test_model_provider():

    provider = (
        get_model_provider()
    )

    print("\n")
    print("=" * 70)
    print(
        "TCO COPY CHECKER"
    )
    print(
        "MODEL PROVIDER TEST"
    )
    print("=" * 70)

    print(
        "\nProvider:",
        provider.get_provider_name(),
    )

    print(
        "Vision model:",
        provider.get_vision_model(),
    )

    print(
        "Text model:",
        provider.get_text_model(),
    )

    print(
        "Base URL:",
        provider.get_base_url(),
    )

    print(
        "\nCapabilities:"
    )

    print(
        json.dumps(
            provider.get_capabilities(),
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    health = (
        provider.health_check()
    )

    print(
        "\nHealth:"
    )

    print(
        json.dumps(
            health,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert (
        provider.get_provider_name()
    )

    assert (
        provider.get_vision_model()
    )

    assert (
        provider.get_text_model()
    )

    assert (
        isinstance(
            health,
            dict,
        )
    )

    print("\n")
    print("=" * 70)
    print(
        "MODEL PROVIDER TEST PASSED"
    )
    print("=" * 70)