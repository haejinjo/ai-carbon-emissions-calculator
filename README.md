# AI Carbon Index

_Easy CO₂ report for AI inference and training, built like AWS Billing._

_Measure emissions from AI operations in real time using public WattTime emissions data, with precision tied to actual usage, infrastructure, and grid impact. Make smarter choices about when, where, and how you deploy AI._

## Why I Built This

AI carbon emissions should be as visible and actionable as cloud costs. Not buried in a once-a-year ESG report that nobody reads — but surfaced in real-time to consumers and executives, right where decisions are made.

I want to see costs appear on every QBR deck, every LLM chatbox, every model marketplace listing — the same way Google shows carbon impact for flights before you book them.

Inspired by [Ramp's AI Index](https://ramp.com/data/ai-index), which visualizes how companies spend on AI — I wanted to know the environmental cost behind that spend.

> How much carbon does all this AI inference actually emit — in real time, at scale?

The answers are out there, but nobody seems to know them.
So I built an agent-powered form that makes it easy by hitting this code, which uses real public data, cloud hardware specs, and simple math formulas to fetch you your answer.

Wanna chat about putting this in production? Send me a message on [LinkedIn](https://www.linkedin.com/in/haejinjo/).

If we're serious about a future for our children, we need to make carbon emissions impossible to ignore.
Thanks for caring about this stuff. I believe we can make carbon-aware AI the default — if we make it as easy and transparent as checking AWS billing.

— Hejin 🌱

---

## What It Does

With just **5 inputs**, the calculator estimates how much CO₂ your AI workloads emit — in real-time.

Inputs:
1. Model Size         → small, medium, large, xlarge
2. Tokens Processed   → e.g. 1,000,000,000
3. Instance Type      → A100, H100, TPUv4, etc.
4. Region             → state or cloud region (e.g. "CA", "us-east-1")
5. Cloud Provider     → AWS, GCP, Azure, or on-prem

Output:

Total emissions in kg CO₂e and Energy usage in kWh.

Example:
```json
{
  "total_emissions_kgCO2e": 2.5,
  "energy_consumption_kWh": 11.3,
  "compute_hours": 3.8,
  "carbon_intensity_gCO2_per_kWh": 203.5,
  "pue_factor": 1.135,
  "power_consumption_watts": 400,
  "emissions_per_million_tokens": 0.011
}
```

## How It Works

Everything is based on this fundamental formula, using the best available data to fill in the gaps:

`Emissions = Power × Time × Carbon Intensity × PUE / 1000`


## Try It Locally

Clone the repo and install dependencies:
```bash
git clone https://github.com/haejinjo/ai-emissions-calculator.git
cd ai-emissions-calculator
pip install -r requirements.txt
python ai_emissions_calculator.py
```

Run the calculator directly with example inputs:
`python ai_emissions_calculator.py`

Or spin up a lightweight REST API (e.g., for integrating into tools, agents, or dashboards):
`uvicorn api:app --reload`
This lets you hit /estimate_emissions with a JSON payload and get back detailed emissions output. Great for internal tools, experimentation, or plugging into your AI stack.

## Tests

This isn't a black box. You can literally walk through every assumption I make with tests.

#### ✅ An accuracy test suite you can run yourself
File: `test_published_benchmarks()` in `test_accuracy.py`

The `test_accuracy.py` script includes a function that simulates known real-world model training workloads — like GPT-3 (~552,000 kg CO₂ for 1000 V100s over 30 days).
It runs your calculator’s logic and compares the output to published results, showing whether you're in a reasonable range (e.g., within 2x–3x of OpenAI’s paper).

#### ✅ Manual sanity checks for energy math
File: `test_energy_calculation()` in `test_accuracy.py`

This test verifies that the core energy calculation (Watts × hours × carbon intensity × PUE) holds up with known constants.

It runs example workloads like:

> “A100 at 65% utilization for 1 hour”
>
> “T4 inference for 10 hours”

Then it compares your calculator’s output against expected kWh using simple arithmetic — and fails the test if the difference is over 5%.

#### ✅ Regional comparisons (Texas vs Oregon vs New York)
File: `test_regional_variations()` in `test_accuracy.py`

Same workload. Different grid. This test runs identical inference jobs across multiple U.S. regions with different carbon intensities — and confirms the calculator reflects expected CO₂ differences.
It shows, for example, that running in Missouri (675 gCO₂/kWh) emits ~5x more than in Upstate NY (129 gCO₂/kWh).

#### ✅ Built-in benchmark validator for real-world scenarios
File: `validate_against_benchmark()` in `ai_emissions_calculator.py`

Inside the core calculator (`AIEmissionsCalculator.validate_against_benchmark()`), you’ll find hardcoded CO₂ benchmarks from well-known AI models:

- GPT-3 training: ~552,000 kg CO₂
- BLOOM training: ~24,700 kg CO₂
- BERT training: ~22.7 kg CO₂
- GPT-3 inference: ~0.4 kg CO₂ per million tokens
- BERT inference: ~0.002 kg CO₂ per million tokens

You can pass your own emissions output into this function and compare it directly to these reference models — so your team can tell if their job was abnormally high or low.

# How You Can Contribute
I’d love your help making this more accurate, useful, and widely adopted.

## Research support
- Better benchmarks for LLM inference throughput
- More granular PUE data by region/provider
- Incorporate batch size scaling data

## More ways to interface
- UI dashboard / web wrapper for accessibility (WIP)
  - Integration to roll up and surface token consumption by team, model, and vendor quarterly
  - Developer SDK & Logging Middleware
Build a lightweight SDK or drop-in middleware for OpenAI, Anthropic, or other LLM APIs that automatically logs token usage per request, attaches metadata (user, team, endpoint), and sends structured logs to popular observability platforms (e.g. Datadog, BigQuery, or AWS CloudWatch). This would let developers track emissions effortlessly with zero manual instrumentation.
  -  Enable businesses to retroactively estimate their AI token usage by parsing invoice costs and reverse-calculating token volumes based on model-specific pricing. This would allow teams without internal logging to generate rough carbon impact estimates using only billing data.
- Optimization CTAs (WIP):
  > "Run this job in Oregon, cut emissions by 35%"
  >
  > “10M queries = 25 kg CO₂e (driving 62 miles). Run this at night in CA and save 20%.”)
- Slack bot: “/carbon ai run”
- GitHub Action for model training pipelines
- Carbon budgeting tool for CFOs or ESG teams.
- Add support for more instance types (e.g., L40S, MI300)
- Integrate GPU utilization logs from cloud billing APIs
- Convert to pip package or Hugging Face Space
- Surface tests via API or CLI to give quick sanity checks like: “Your emissions are 1.2x that of BLOOM training. That’s expected for a similar workload.”

Just open a PR or an issue — or email me if you want to jam.

## Feedback & Community

This is just v0. 

If you see something that’s off, have better data, or want to adapt this for your org — I’m all ears. The goal isn’t to be right — it’s to make it easier to be right when it matters.

# License
MIT — use, remix, credit, and deploy freely.

# Resources
- [AI’s Growing Carbon Footprint, Columbia University, June 2023](https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/)
- [Yes, AI Has a Carbon Footprint, Vice, June 2019](https://www.vice.com/en/article/training-one-ai-model-produces-as-much-emissions-as-a-cross-country-flight-study-finds/)
- See how Google calculates flight carbon emissions with their [open source Travel Impact Model](https://travelimpactmodel.org/)
- Found an equivalent Python integration: https://codecarbon.io/