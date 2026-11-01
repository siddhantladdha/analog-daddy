"""
Clanker written tests.

- Created/Changed using: Claude Code
- HUMAN VERIFICATION: DONE
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from analog_daddy.units import (
    Capacitance,
    Current,
    Frequency,
    Inductance,
    Power,
    Resistance,
    Temperature,
    Time,
    Voltage,
)


class _M(BaseModel):
    time: Time


# --- parsing / coercion -----------------------------------------------------


def test_explicit_units_string():
    t = Time("1 ns")
    assert t.units == "s"
    assert float(t) == pytest.approx(1e-9)


def test_bare_si_prefix_string_attaches_declared_units():
    # the whole point of subclassing: "1n" == "1 ns" for a Time field
    assert Time("1n") == Time("1 ns")
    assert Time("1n").units == "s"


def test_bare_float_attaches_declared_units():
    t = Time(1e-9)
    assert t.units == "s"
    assert float(t) == pytest.approx(1e-9)


def test_bare_unitless_string_attaches_declared_units():
    t = Time("1")
    assert t.units == "s"
    assert float(t) == 1.0


@pytest.mark.parametrize(
    ("cls", "units"),
    [
        (Time, "s"),
        (Voltage, "V"),
        (Current, "A"),
        (Power, "W"),
        (Resistance, "Ohm"),
        (Capacitance, "F"),
        (Inductance, "H"),
        (Temperature, "C"),
        (Frequency, "Hz"),
    ],
)
def test_every_subclass_attaches_its_declared_units(cls, units):
    q = cls("1")
    assert q.units == units


# --- unit conversion / aliasing ---------------------------------------------


@pytest.mark.parametrize("text", ["5 Ω", "5 ohm", "5 ohms"])
def test_ohm_symbol_aliases_normalize_to_declared_units(text):
    r = Resistance(text)
    assert r.units == "Ohm"
    assert float(r) == 5.0


def test_kilo_ohm_still_scales_normally_after_alias_registered():
    r = Resistance("5 kOhm")
    assert r.units == "Ohm"
    assert float(r) == 5000.0


# --- validation errors --------------------------------------------------


def test_incompatible_units_raise_pydantic_validation_error_not_raw_quantiphy_error():
    # QuantiPhyError (here UnknownConversion, a KeyError subclass) must be
    # wrapped as ValueError in _parse, or pydantic-core won't catch it and
    # this raises an unhandled quantiphy exception instead of ValidationError.
    with pytest.raises(ValidationError) as exc_info:
        _M(time="5 kg")
    errors = exc_info.value.errors()
    # union_schema (is_instance_schema + plain_validator) surfaces BOTH
    # branch failures -- this is expected/accepted behavior, not a bug.
    assert len(errors) == 2
    assert errors[0]["type"] == "is_instance_of"
    assert errors[1]["type"] == "value_error"
    assert "unable to convert" in errors[1]["msg"]


def test_existing_instance_passes_through_is_instance_branch_by_identity():
    existing = Time("5 ns")
    m = _M(time=existing)
    # is_instance_schema branch should return the SAME object, not a copy.
    assert m.time is existing


def test_bare_number_is_accepted_not_rejected():
    # no strict mode: bare numbers are the intended shorthand, not an error.
    m = _M(time=1e-9)
    assert m.time.units == "s"


# --- serialization ------------------------------------------------------


def test_default_serialization_is_as_real():
    m = _M(time="1n")
    assert m.model_dump_json() == '{"time":1e-9}'


def test_quantity_format_str_context():
    m = _M(time="1n")
    dumped = m.model_dump_json(context={"quantity_format": "str"})
    assert dumped == '{"time":"1 ns"}'


def test_quantity_format_tuple_context():
    m = _M(time="1n")
    dumped = m.model_dump_json(context={"quantity_format": "as_tuple"})
    assert dumped == '{"time":[1e-9,"s"]}'


def test_serialize_unknown_mode_raises():
    m = _M(time="1n")
    with pytest.raises(Exception, match="Unknown quantity_format value"):
        m.model_dump_json(context={"quantity_format": "bogus"})


def test_python_mode_dump_currently_loses_subclass_identity():
    # KNOWN GAP: when_used="json" should, per pydantic docs, leave
    # python-mode model_dump() untouched (keep the Time instance). In
    # practice it degrades to a plain float, because Time subclasses
    # float and pydantic-core's default python-mode fallback flattens it.
    # This test documents CURRENT behavior, not desired behavior.
    m = _M(time="1n")
    dumped = m.model_dump()
    assert type(dumped["time"]) is float
    assert not isinstance(dumped["time"], Time)


# --- round-trips ----------------------------------------------------------


def test_default_as_real_round_trips():
    m = _M(time="1n")
    m2 = _M.model_validate_json(m.model_dump_json())
    assert float(m2.time) == pytest.approx(float(m.time))


def test_as_tuple_round_trip_currently_fails():
    # KNOWN GAP: _parse only handles float | str, so the [value, units]
    # shape produced by quantity_format="as_tuple" cannot be re-validated.
    m = _M(time="1n")
    dumped = m.model_dump_json(context={"quantity_format": "as_tuple"})
    with pytest.raises(ValidationError, match="not a valid number"):
        _M.model_validate_json(dumped)


# --- JSON schema ------------------------------------------------------


def test_json_schema_has_description_and_examples_not_empty_anyof():
    schema = _M.model_json_schema()
    t_schema = schema["properties"]["time"]
    assert t_schema["type"] == "string"
    assert "description" in t_schema
    assert t_schema["examples"]  # non-empty
    assert "anyOf" not in t_schema  # regression check for the original bug


# --- property-based tests (hypothesis) --------------------------------------

# quantiphy's SI prefixes span 1e-24 (y) .. 1e24 (Y) -- bound the strategy to
# that range so generated values stay renderable.
reasonable_floats = st.floats(
    min_value=-1e24, max_value=1e24, allow_nan=False, allow_infinity=False
)

# lets hypothesis auto-generate Time instances for any model that has a
# Time field, e.g. via st.from_type(SomeModel).
st.register_type_strategy(Time, reasonable_floats.map(Time))


@given(reasonable_floats)
@settings(max_examples=200)
def test_bare_float_round_trips_through_default_serialization(x):
    m = _M(time=x)
    assert m.time.units == "s"
    m2 = _M.model_validate_json(m.model_dump_json())
    assert float(m2.time) == float(m.time)


@given(reasonable_floats)
@settings(max_examples=200)
def test_existing_instance_identity_always_preserved(x):
    existing = Time(x)
    m = _M(time=existing)
    assert m.time is existing  # is_instance_schema branch, never reconstructed


@given(st.from_type(_M))
@settings(max_examples=50)
def test_any_generated_model_round_trips(m):
    # once register_type_strategy is set, hypothesis can build whole
    # models automatically, not just the Time field in isolation.
    m2 = _M.model_validate_json(m.model_dump_json())
    assert float(m2.time) == float(m.time)
