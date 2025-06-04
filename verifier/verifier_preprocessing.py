import pandas as pd
import re

# Process the file line by line in chunks
chunks = pd.read_json("openmathreasoning/tir.jsonl", lines=True, chunksize=1000)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}")
    print(chunk.head())
    print(chunk.columns)
    if i == 0:
        break  # Only print the first chunk for now


# Initialize empty list to collect filtered samples
filtered_rows = []

for chunk in chunks:
    filtered_chunk = chunk[chunk['problem_type'] == "has_answer_extracted"]
    filtered_rows.append(filtered_chunk)
    
    # Stop early if we’ve collected enough
    total_rows = sum(len(df) for df in filtered_rows)
    if total_rows >= 11000:  # 10k train + 1k test
        break

# Concatenate and shuffle
filtered_df = pd.concat(filtered_rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Take only the first 11,000 after shuffling
filtered_df = filtered_df.head(11000)

# Split into train/test
train_df = filtered_df.iloc[:10000]
test_df = filtered_df.iloc[10000:]

# Save to JSONL
train_df.to_json("openmathreasoning/train_filtered.jsonl", orient="records", lines=True)
test_df.to_json("openmathreasoning/test_filtered.jsonl", orient="records", lines=True)


def extract_think_paragraph_and_tool_calls(text):
    think_paragraph = ""
    tool_calls = []

    # Extract the first paragraph starting with <think>
    think_match = re.search(r"(<think>.*?)(?:\n\s*\n|$)", text, re.DOTALL)
    if think_match:
        think_paragraph = think_match.group(1).strip()

    # Extract all <tool_call>...</tool_call> blocks
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    tool_output = "\n\n".join(tc.strip() for tc in tool_calls)

    return think_paragraph, tool_output


# Load filtered dataset
df = pd.read_json("openmathreasoning/test_filtered.jsonl", lines=True)

# Extract <think> paragraph and <tool_call> blocks
df[["think_paragraph", "tool_output"]] = df["generated_solution"].apply(
    lambda x: pd.Series(extract_think_paragraph_and_tool_calls(x))
)

# Build instruction-style input
df["input"] = (
    "You are a reasoning assistant. Given the math problem and the beginning of a thought process, "
    "generate the code that would be used to solve it.\n\n"
    "Problem:\n" + df["problem"].str.strip() + "\n\n" + df["think_paragraph"].str.strip()
)

# Wrap output in <code> tags
df["output"] = "<code>\n" + df["tool_output"].str.strip() + "\n</code>"

# Prepare final dataset
final_df = df[["input", "output", "expected_answer"]]

# Drop empty rows
final_df = final_df[
    (final_df["input"].str.strip() != "") &
    (final_df["output"].str.strip() != "")
]

# Save to JSONL
final_df.to_json("openmathreasoning/test_toolcall_model_instruct.jsonl", orient="records", lines=True)