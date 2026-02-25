"""
Energy Knowledge Base
======================
Pre-written documents about ERCOT and Texas energy markets.

WHY THESE DOCUMENTS EXIST:
    The RAG service needs documents to search through.
    In production, you'd ingest real ERCOT notices, EIA reports, and news.
    These sample documents provide a realistic knowledge base for demo
    and testing, covering common scenarios the trading bot would encounter.

    When deployed, the service also fetches real energy news via the
    /ingest/news endpoint. These samples ensure the RAG works
    even without internet access.
"""

ENERGY_DOCUMENTS = [
    {
        "text": """ERCOT Conservation Alert: When ERCOT issues a conservation alert, 
it signals that grid conditions are tight and electricity demand is approaching 
available supply. This typically occurs during extreme heat events when air 
conditioning demand surges across Texas. Conservation alerts often precede 
price spikes, as generators bid higher prices when supply margins thin. 
During past conservation events, HB_NORTH settlement point prices have 
exceeded $200/MWh, with some intervals reaching $1,000+/MWh. Traders 
should consider reducing buy positions and increasing sell orders when 
conservation alerts are active, as prices typically spike within 2-6 hours 
of the alert.""",
        "metadata": {"source": "ercot_knowledge", "topic": "conservation_alert"},
    },
    {
        "text": """Generator Outages and ERCOT Prices: Unplanned generator outages 
are a major driver of price spikes in ERCOT. Texas has approximately 90GW of 
installed generation capacity. When a large plant (500MW+) trips offline 
unexpectedly, the supply-demand balance tightens immediately. Nuclear plants 
like Comanche Peak (2,430MW total) and South Texas Project (2,710MW) are 
particularly impactful because they provide baseload power. If one unit at 
Comanche Peak goes offline (1,215MW), it removes about 1.3% of total grid 
capacity. Combined with high demand days, this can push prices from normal 
levels ($30-50/MWh) to extreme levels ($500-5,000/MWh). The 2021 Winter Storm 
Uri caused cascading generator failures that pushed prices to the $9,000/MWh 
cap for several consecutive days.""",
        "metadata": {"source": "ercot_knowledge", "topic": "generator_outage"},
    },
    {
        "text": """Texas Wind Energy and Price Impact: Texas leads the US in wind 
energy generation, with approximately 40GW of installed wind capacity as of 2024. 
Wind farms are concentrated in West Texas (Permian Basin) and the Texas Panhandle. 
Wind generation follows predictable patterns: output is typically highest at night 
and during spring/fall, and lowest during hot summer afternoons when the atmosphere 
is stable. When wind output drops below 10GW (25% of capacity), it creates upward 
pressure on prices because gas plants must fill the gap at higher marginal costs. 
Conversely, when wind exceeds 25GW, prices can drop to near-zero or even go 
negative as excess supply floods the market. HB_WEST hub prices are most directly 
affected by wind output changes.""",
        "metadata": {"source": "ercot_knowledge", "topic": "wind_energy"},
    },
    {
        "text": """Texas Solar Energy Growth: Solar capacity in ERCOT has grown 
rapidly, reaching approximately 20GW by 2024. Solar generation peaks between 
11am-3pm CST and drops to zero at sunset. This creates the "duck curve" pattern 
where net demand (total demand minus solar) dips during midday and surges in 
the evening as solar disappears and people return home. The evening ramp 
(typically 5pm-8pm) is becoming increasingly steep as more solar is added, 
creating price volatility during these hours. During clear summer days, solar 
can supply 15-20% of total ERCOT demand, significantly depressing midday prices. 
Cloud cover events can cause rapid swings in solar output, leading to 
intra-hour price volatility.""",
        "metadata": {"source": "ercot_knowledge", "topic": "solar_energy"},
    },
    {
        "text": """Natural Gas Prices and Electricity Costs: Natural gas fuels 
approximately 45% of ERCOT generation. When natural gas prices rise, the marginal 
cost of gas-fired generation increases, pushing electricity prices higher. The 
relationship is roughly: a $1/MMBtu increase in gas prices translates to a 
$7-10/MWh increase in electricity prices. Key benchmarks include Henry Hub 
(national) and Waha Hub (West Texas, often negative due to pipeline constraints). 
When Waha prices disconnect from Henry Hub, it signals local supply-demand 
imbalances that can affect ERCOT west zone prices. Seasonal gas storage levels 
also matter: low storage heading into summer increases price spike risk because 
generators face higher fuel costs during peak demand periods.""",
        "metadata": {"source": "ercot_knowledge", "topic": "natural_gas"},
    },
    {
        "text": """ERCOT Demand Patterns: Texas electricity demand is highly 
seasonal and weather-driven. Summer peak demand can exceed 80GW during heat 
waves, while winter demand typically peaks around 60-65GW. The primary demand 
driver is air conditioning, which accounts for roughly 30-40% of total summer 
demand. Each degree above 95°F (35°C) adds approximately 2-3GW of additional 
AC demand. Demand is also influenced by economic activity (industrial load), 
time of day (commercial hours vs residential evening), and day of week 
(weekdays higher than weekends). ERCOT publishes its own demand forecasts, 
which traders use as a benchmark. When actual demand exceeds ERCOT's forecast, 
prices tend to spike as the market was not positioned for the higher load.""",
        "metadata": {"source": "ercot_knowledge", "topic": "demand_patterns"},
    },
    {
        "text": """Extreme Weather Events in Texas: Texas is prone to extreme 
weather that severely impacts the grid. Summer heat waves (105°F+) push demand 
to record levels while reducing thermal plant efficiency. Winter storms can 
freeze gas wells and pipelines, cutting fuel supply to generators. Hurricane 
season (June-November) threatens Gulf Coast generation and transmission 
infrastructure. These events create the highest-risk trading periods. During 
Winter Storm Uri (February 2021), prices hit the $9,000/MWh cap for over 
100 consecutive hours, with total market costs exceeding $47 billion in one 
week. The lesson: extreme weather events in ERCOT are not black swans — they 
are predictable seasonal risks that should be factored into trading strategy.""",
        "metadata": {"source": "ercot_knowledge", "topic": "extreme_weather"},
    },
    {
        "text": """ERCOT Real-Time Market Structure: ERCOT operates a nodal 
wholesale electricity market with two key pricing mechanisms: Day-Ahead Market 
(DAM) and Real-Time Market (RTM). The DAM settles hourly based on bids 
submitted the day before. The RTM settles every 15 minutes based on actual 
conditions. Our trading system operates in the RTM timeframe, making decisions 
every 15 minutes aligned with settlement intervals. Key price points include 
settlement point prices (SPP) at hub and load zone levels. The four main hubs 
are HB_NORTH (Dallas), HB_SOUTH (San Antonio), HB_HOUSTON, and HB_WEST. 
Price differences between hubs indicate transmission congestion — when 
HB_WEST prices are much lower than HB_NORTH, it means cheap wind energy 
in West Texas cannot reach the Dallas demand center.""",
        "metadata": {"source": "ercot_knowledge", "topic": "market_structure"},
    },
]