import faust
from typing import Optional
from statistics import mean, stdev

app = faust.App(
    'taxi-stream-app',
    broker='kafka://broker1-kr:9092',
    value_serializer='json',
)

# Schema of incoming messages
class TaxiRecord(faust.Record, serializer='json'):
    total_amount: Optional[float]
    passenger_count: Optional[int]

# Schema for outgoing messages
class StatsRecord(faust.Record, serializer='json'):
    mean_fare: float
    std_fare: float
    mean_psg: float
    std_psg: float
    count: int



# Topic from your producer
taxi_topic = app.topic('yellow_taxi_stream', value_type=TaxiRecord)
stats_topic = app.topic('yellow_taxi_stats', value_serializer='json')


# Table for stats
stats_table = app.Table(
    'total_stats',
    default=lambda: {
        'count': 0,
        'total_amounts': [],
        'passengers': []
    },
    partitions=1,
    changelog_topic=app.topic('custom_stats_changelog', partitions=1)
)


@app.agent(taxi_topic)
async def process(taxis):
    async for taxi in taxis:
        borough = taxi.pickup_borough
        print(taxi.total_amount)
        stats = stats_table[borough]  

        # Update stats
        stats['count'] += 1
        stats['total_amounts'].append(taxi.total_amount or 0)
        stats['passengers'].append(taxi.passenger_count or 0)

        # Maintain a rolling window of last 100 values
        for key in ['total_amounts', 'passengers']:
            if len(stats[key]) > 100:
                stats[key].pop(0)
                
        stats_table[borough] = stats
        # Calculate and print rolling mean and std
        if len(stats['total_amounts']) > 1:  # stdev needs at least 2 values
            mean_fare = mean(stats['total_amounts'])
            std_fare = stdev(stats['total_amounts'])
            mean_psg = mean(stats['passengers'])
            std_psg = stdev(stats['passengers'])
            count = stats['count']
            
            print(f"{borough} Count={stats['count']}")
            print(f"  💰 Mean Fare: {mean_fare:.2f}, Std: {std_fare:.2f}")
            print(f"  👥 Mean Passengers: {mean_psg:.2f}, Std: {std_psg:.2f}")

                
            stats_msg = StatsRecord(
                mean_fare=mean_fare,
                std_fare=std_fare,
                mean_psg=mean_psg,
                std_psg=std_psg,
                count=count
            )
    
            await stats_topic.send(value=stats_msg)
        

