"""
Pydantic ready ``quantiphy.Quantity`` units

Examples
--------

>>> from pydantic import BaseModel
>>> class RC(BaseModel):
...     resistance: Resistance
...     time: Time
>>> rc = RC(resistance="4.7 kOhm", time="1n")
>>> rc.resistance
Resistance('4.7 kOhm')
>>> rc.time
Time('1 ns')
>>> rc.model_dump_json()
'{"resistance":4700.0,"time":1e-9}'
>>> rc.model_dump_json(context={"quantity_format": "str"})
'{"resistance":"4.7 kOhm","time":"1 ns"}'
>>> rc.model_dump_json(context={"quantity_format": "as_tuple"})
'{"resistance":[4700.0,"Ohm"],"time":[1e-9,"s"]}'

Notes
-----

The following approaches were considered based on Pydantic docs and
discussions with a LLM model for parts which confused me.

Chosen approach: Put the pydantic hook on the type itself. Works
everywhere automatically, best autocomplete, but ties Time to pydantic
and uses a lower-level API pydantic calls experimental.

- Annotate as str, secretly swap it to Time in a validator.
Rejected since: the type hint lie, so
autocomplete/type-checker/serialization all get confused.
- model_validator(mode="after"). Only makes sense when one field needs to be derived from
another (cross-field logic), not for a single field's own conversion.
- Approach from pydantic_pint seemed unncessary, but might have to consider if
there are any issues.
"""

from typing import TYPE_CHECKING, Any, Self

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    SerializationInfo,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from quantiphy import (
    QuantiPhyError,
    UnitConversion,
    as_real,
    as_tuple,
)
from quantiphy import Quantity as _Quantity

UnitConversion(["ohm", "Ohms", "ohms", "Ω"], "Ohm")


class Quantity(_Quantity):
    """
    Wrapper around ``quantiphy.Quantity`` to add Pydantic support.
    """

    units: str = ""

    if TYPE_CHECKING:

        def __new__(
            cls,
            value: float | str | Self,
            *args: Any,
            **kwargs: Any,
        ) -> Self: ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Generate a Pydantic core schema for the Quantity type.

        The schema is union based with the serialization support for
        as_real, str, and as_tuple serialization options using the
        context dictionary.
        """
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.no_info_plain_validator_function(cls._parse),
            ],
            mode="left_to_right",
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=True,
                return_schema=core_schema.union_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.float_schema(),
                        core_schema.tuple_schema(
                            items_schema=[
                                core_schema.float_schema(),
                                core_schema.str_schema(),
                            ],
                            strict=True,
                        ),
                    ]
                ),
            ),
        )

    @classmethod
    def _parse(cls, value: float | str) -> Self:
        """
        Parse value using Quantiphy's native mechanism.
        """
        try:
            return cls(value)
        except QuantiPhyError as error:
            # TODO: Consider custom error message for user.
            raise ValueError(str(error)) from error

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate a JSON schema for the Quantity type."""
        json_schema = handler(core_schema.str_schema())
        json_schema["description"] = (
            f"Physical quantity with units: {cls.units}"
            if cls.units
            else "Quantiphy quantity without units"
        )
        json_schema["examples"] = (
            [f"1 {cls.units}", f"1 n{cls.units}", f"1 M{cls.units}"]
            if cls.units
            else ["1", "1n", "1M"]
        )
        return json_schema

    def _serialize(
        self, info: SerializationInfo
    ) -> str | float | tuple[float, str]:
        """Customizable options to serialize the quantity."""
        if isinstance(info.context, dict):
            quantity_format = info.context.get("quantity_format", "as_real")
            match quantity_format:
                case "as_real":
                    return as_real(self)
                case "str":
                    return str(self)
                case "as_tuple":
                    return as_tuple(self)
                case _:
                    raise ValueError(
                        f"Unknown quantity_format value: {quantity_format}"
                    )
        else:
            return as_real(self)


class Time(Quantity):
    """
    Time quantity with units of seconds.

    Attributes
    ----------
    units : str, default="s"
        Unit symbol.

    Examples
    --------
    >>> Time("1 ns")
    Time('1 ns')
    >>> Time(1e-9)
    Time('1 ns')
    """

    units: str = "s"


class Voltage(Quantity):
    """
    Voltage quantity with units of Volts.

    Attributes
    ----------
    units : str, default="V"
        Unit symbol.

    Examples
    --------
    >>> Voltage("1 uV")
    Voltage('1 uV')
    >>> Voltage(1e-9)
    Voltage('1 nV')
    """

    units: str = "V"


class Current(Quantity):
    """
    Current quantity with units of Amperes.

    Attributes
    ----------
    units : str, default="A"
        Unit symbol.

    Examples
    --------
    >>> Current("1 uA")
    Current('1 uA')
    >>> Current(1e-9)
    Current('1 nA')
    """

    units: str = "A"


class Power(Quantity):
    """
    Power quantity with units of Watts.

    Attributes
    ----------
    units : str, default="W"
        Unit symbol.

    Examples
    --------
    >>> Power("12 mW")
    Power('12 mW')
    >>> Power(1e3)
    Power('1 kW')
    """

    units: str = "W"


class Resistance(Quantity):
    """
    Resistance quantity with units of Ohms/ohm/etc.

    Attributes
    ----------
    units : str, default="Ohm"
        Unit symbol.

    Examples
    --------
    >>> Resistance("5 ohms")
    Resistance('5 Ohm')
    >>> Resistance("5 mΩ")
    Resistance('5 mOhm')
    >>> Resistance(50)
    Resistance('50 Ohm')
    """

    units: str = "Ohm"


class Capacitance(Quantity):
    """
    Capacitance quantity with units of Farads.

    Attributes
    ----------
    units : str, default="F"
        Unit symbol.

    Examples
    --------
    >>> Capacitance("5e-15 F")
    Capacitance('5 fF')
    """

    units: str = "F"


class Inductance(Quantity):
    """
    Inductance quantity with units of Henry.

    Attributes
    ----------
    units : str, default="H"
        Unit symbol.

    Examples
    --------
    >>> Inductance(5e-9)
    Inductance('5 nH')
    """

    units: str = "H"


class Temperature(Quantity):
    """
    Temperature quantity with units of Celsius.

    Attributes
    ----------
    units : str, default="C"
        Unit symbol.

    Examples
    --------
    >>> Temperature(25)
    Temperature('25 C')
    """

    units: str = "C"


class Frequency(Quantity):
    """
    Frequency quantity with units of Hertz.

    Attributes
    ----------
    units : str, default="Hz"
        Unit symbol.

    Examples
    --------
    >>> Frequency(5e9)
    Frequency('5 GHz')
    """

    units: str = "Hz"
