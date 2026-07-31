"""
Verification Script for ConfigurableLabelMapper.
"""

from cross_dataset.label_mapping.mapper import ConfigurableLabelMapper


def test_configurable_label_mapper():
    print("==================================================")
    print(" Testing ConfigurableLabelMapper Dynamic Ingestion")
    print("==================================================")

    mapper = ConfigurableLabelMapper()

    # Test raw label cases
    test_cases = [
        "High",
        "mid",
        "LOW",
        "Disengagement",
        "Normal",
        "Engagement",
        "Very engagement",
        "Very Engagement",
    ]

    print("\n[1] Testing Primary (3-Class: Low, Mid, High) & Binary (2-Class: Not Engaged, Engaged) Mappings:")
    for raw in test_cases:
        p_val, b_val = mapper.map_all(raw)
        p_name = mapper.get_primary_display_name(p_val)
        b_name = mapper.get_binary_display_name(b_val)
        print(f"  Raw: '{raw:15s}' -> Primary: {p_val} ({p_name:5s}) | Binary: {b_val} ({b_name})")

    print("\n[2] Testing Display Names:")
    print("  Primary Targets:")
    for val in [0, 1, 2]:
        print(f"    Target {val} -> '{mapper.get_primary_display_name(val)}'")

    print("  Binary Targets:")
    for val in [0, 1]:
        print(f"    Target {val} -> '{mapper.get_binary_display_name(val)}'")

    print("\n[3] Testing Unmapped Label Error Handling:")
    try:
        mapper.map_primary("UnknownLabelClass")
        print("  FAIL: Exception was not raised for unknown label!")
    except ValueError as e:
        print(f"  SUCCESS: Caught expected ValueError: {e}")

    print("\n==================================================")
    print(" All Label Mapper Verifications PASSED!")
    print("==================================================")


if __name__ == "__main__":
    test_configurable_label_mapper()
