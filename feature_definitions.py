from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float32, Int64
from feast.value_type import ValueType


# ============================================================
# PROJECT
# ============================================================

project = Project(
    name="adequate_stud",
    description="Feast feature store for Lahore AQI prediction",
)


# ============================================================
# ENTITY
# ============================================================

aqi_location = Entity(
    name="aqi_location",
    join_keys=["location_id"],
    value_type=ValueType.INT64,
    description="Location identifier for Lahore AQI prediction",
)


# ============================================================
# DATA SOURCE
# ============================================================

aqi_source = FileSource(
    name="lahore_aqi_source",
    path="data/lahore_features.parquet",
    timestamp_field="event_timestamp",
    description="Lahore air quality and weather features",
)


# ============================================================
# FEATURE VIEW
# ============================================================

aqi_features = FeatureView(
    name="lahore_air_quality_features",

    entities=[aqi_location],

    ttl=timedelta(days=7),

    schema=[
        # ----------------------------------------------------
        # WEATHER FEATURES
        # ----------------------------------------------------

        Field(
            name="temperature_2m",
            dtype=Float32,
        ),

        Field(
            name="relative_humidity_2m",
            dtype=Float32,
        ),

        Field(
            name="precipitation",
            dtype=Float32,
        ),

        Field(
            name="surface_pressure",
            dtype=Float32,
        ),

        Field(
            name="wind_speed_10m",
            dtype=Float32,
        ),

        Field(
            name="wind_direction_10m",
            dtype=Float32,
        ),

        # ----------------------------------------------------
        # AIR QUALITY FEATURES
        # ----------------------------------------------------

        Field(
            name="pm10",
            dtype=Float32,
        ),

        Field(
            name="pm2_5",
            dtype=Float32,
        ),

        Field(
            name="carbon_monoxide",
            dtype=Float32,
        ),

        Field(
            name="nitrogen_dioxide",
            dtype=Float32,
        ),

        Field(
            name="sulphur_dioxide",
            dtype=Float32,
        ),

        Field(
            name="ozone",
            dtype=Float32,
        ),

        Field(
            name="dust",
            dtype=Float32,
        ),

        # ----------------------------------------------------
        # TIME FEATURES
        # ----------------------------------------------------

        Field(
            name="hour",
            dtype=Int64,
        ),

        Field(
            name="day",
            dtype=Int64,
        ),

        Field(
            name="day_of_week",
            dtype=Int64,
        ),

        Field(
            name="month",
            dtype=Int64,
        ),

        Field(
            name="year",
            dtype=Int64,
        ),

        Field(
            name="is_weekend",
            dtype=Int64,
        ),
    ],

    online=True,

    source=aqi_source,

    tags={
        "team": "aqi_prediction",
        "project": "lahore_aqi",
        "environment": "development",
    },
)