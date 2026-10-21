"""
Station metadata and the documented history of each record.

The Padova history is taken from Marani & Zanetti (2015, Water Resources
Research, 51, 639-647, section 2.1), which follows Camuffo (1984). It is used
for two things: to shade the observational eras in figures, and to interpret
quality-control flags. It is never used to alter data values.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Era:
    """A documented observational period of a station record."""
    start: int
    end: int
    short: str            # label drawn on figures
    description: str      # full description for reports


@dataclass(frozen=True)
class Station:
    key: str
    name: str
    source: str
    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    raw_unit: str
    eras: tuple[Era, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


PADOVA_ERAS: tuple[Era, ...] = (
    Era(1725, 1767, "Poleni",
        "G. Poleni, gauge on the roof of his house (~10 m)."),
    Era(1768, 1813, "Toaldo-Chiminello",
        "G. Toaldo, then V. Chiminello, La Specola observatory (~25 m)."),
    Era(1814, 1877, "Santini",
        "G. Santini, La Specola; from 1838 the whole observatory roof "
        "(27.5 m^2) is used as a funnel, which may under-record small rain "
        "rates."),
    Era(1878, 1934, "Lorenzoni et al.",
        "G. Lorenzoni's raingauge (0.4 m^2), then Antoniazzi and Silva."),
    Era(1935, 1996, "Water Authority",
        "Venice Water Authority station, ~800 m away."),
    Era(1997, 2019, "Bot. Garden",
        "Hourly station at the Padua Botanical Garden, ~1 km away."),
)

STATIONS: dict[str, Station] = {
    "bologna": Station(
        key="bologna",
        name="Bologna",
        source="NOAA NCEI GHCN-Daily, station ITE00100550 (BOLOGNA IT)",
        latitude=44.5000, longitude=11.3458, elevation_m=53.0,
        raw_unit="tenths of mm (GHCN-Daily PRCP convention)",
        eras=(),
        notes=(
            "The extract contains only the PRCP value: the GHCN-Daily "
            "measurement, quality and source flags are not included, so "
            "no upstream QC information is available.",
        ),
    ),
    "padova": Station(
        key="padova",
        name="Padova",
        source=("Padova daily series (Poleni-La Specola), digitised from the "
                "original registers; Marani & Zanetti (2015)"),
        latitude=None, longitude=None, elevation_m=None,
        raw_unit="mm (values before 1864 converted from historical units)",
        eras=PADOVA_ERAS,
        notes=(
            "Marani & Zanetti (2015): 14 years missing (1765-1767, "
            "1815-1822 excluded for sparsity and uncertain/interpolated "
            "values, 1838-1840); 52 isolated missing days interpolated from "
            "neighbouring stations.",
            "Marani & Zanetti (2015): values before 1864 were recorded in "
            "French inches (27.07 mm) and converted to mm.",
            "Marani & Ignaccolo (2015, AWR) use only the uninterrupted "
            "intervals 1725-1764, 1768-1807, 1841-1880 and 1887-2006.",
            "della Valle et al. (2023, Climatic Change 176:9) document, for "
            "1812-1864, irregular reading times and cumulative readings "
            "spanning several rainy days (about 25% cumulative after 1830), "
            "which independently explains the accumulation pattern our QC "
            "flags in 1823-1841. They provide a corrected series.",
            "Camuffo et al. (2020, Climatic Change 162:923-942) stress that "
            "the instrumental threshold controls the frequency of "
            "precipitation days, and use Bologna and Venice as parallel "
            "series.",
            "Hypothesis to verify: 1725-1764 values are exact multiples of "
            "0.0254 mm (1/1000 English inch), whereas Marani & Zanetti (2015) "
            "state 'French inches' before 1864. If they were French inches "
            "converted with the English factor, depths would be about 6.2% "
            "too low (27.07/25.4 = 1.066).",
        ),
    ),
}
