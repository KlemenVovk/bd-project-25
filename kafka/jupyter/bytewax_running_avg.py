from bytewax import operators as op
from bytewax.dataflow import Dataflow
from bytewax.connectors.kafka import KafkaSource
from bytewax.connectors.stdio import StdOutSink
import json

WINDOW_SIZE = 5
brokers = ["broker1-kr:9092"]
flow = Dataflow("rolling_avg_per_borough")

# Kafka input
stream = op.input("in", flow, KafkaSource(brokers, ["yellow_taxi_stream"]))

# Parse and key by borough
keyed = op.key_on("key_by_borough", stream, lambda msg: msg.key.decode("utf-8"))
keyed = op.map_value("parse_json", keyed, lambda msg: json.loads(msg.value.decode("utf-8")))
keyed = op.map_value("get_amount", keyed, lambda msg: float(msg["total_amount"]))

# Rolling average calculation
def rolling_avg(state, new_value):
    if state is None:
        state = []

    print("Before state:", state)
    print("New value:", new_value)

    state.append(new_value)
    if len(state) > WINDOW_SIZE:
        state.pop(0)

    print("After state:", state)
    avg = round(sum(state) / len(state), 2)
    print("Running avg:", avg)
    print()

    return (state, avg)

# Apply rolling average per key
rolling_avgs = op.stateful_map("rolling_avg", keyed, rolling_avg)

# Output to stdout
op.output("print_out", rolling_avgs, StdOutSink())
