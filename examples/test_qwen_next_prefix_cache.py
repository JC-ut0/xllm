# -*- coding: utf-8 -*-
"""
Test script for Qwen-Next Prefix Cache feature.

This script tests the mamba_cache_mode parameter and prefix cache functionality
for Qwen-Next models with GDN (Gated Delta Net) linear attention layers.

Usage:
    # Basic test with align mode (recommended for Qwen-Next)
    python test_qwen_next_prefix_cache.py \
        --model /path/to/Qwen3-Next-80B-A3B-Instruct \
        --mamba_cache_mode align

    # Test with all mode (only for models that support it)
    python test_qwen_next_prefix_cache.py \
        --model /path/to/Qwen3-Next-80B-A3B-Instruct \
        --mamba_cache_mode all

    # Test with none mode (disable mamba state caching)
    python test_qwen_next_prefix_cache.py \
        --model /path/to/Qwen3-Next-80B-A3B-Instruct \
        --mamba_cache_mode none
"""

import argparse
import time
from xllm import LLM, RequestParams


def parse_args():
    parser = argparse.ArgumentParser(description='Test Qwen-Next Prefix Cache')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the Qwen-Next model')
    parser.add_argument('--devices', type=str, default='auto',
                        help='Devices to use (e.g., "cuda:0" or "npu:0")')
    parser.add_argument('--mamba_cache_mode', type=str, default='align',
                        choices=['none', 'all', 'align'],
                        help='Mamba cache mode: none, all, or align')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Block size for KV cache')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    return parser.parse_args()


def test_basic_generation(llm, prompts, request_params):
    """Test basic generation functionality."""
    print("\n" + "="*60)
    print("Test 1: Basic Generation")
    print("="*60)
    
    outputs = llm.generate(prompts, request_params, True)
    
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {output.prompt[:100]}...")
        print(f"Generated: {output.outputs[0].text}")
    
    return outputs


def test_prefix_cache_hit(llm, common_prefix, unique_suffixes, request_params):
    """Test prefix cache hit with shared prefix."""
    print("\n" + "="*60)
    print("Test 2: Prefix Cache Hit Test")
    print("="*60)
    
    prompts = [common_prefix + suffix for suffix in unique_suffixes]
    
    # First run - should miss cache
    print("\nFirst run (cache miss expected)...")
    start_time = time.time()
    outputs1 = llm.generate(prompts, request_params, True)
    first_run_time = time.time() - start_time
    print(f"First run time: {first_run_time:.2f}s")
    
    # Second run with same prompts - should hit cache
    print("\nSecond run (cache hit expected)...")
    start_time = time.time()
    outputs2 = llm.generate(prompts, request_params, True)
    second_run_time = time.time() - start_time
    print(f"Second run time: {second_run_time:.2f}s")
    
    # Calculate speedup
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1.1:
            print("SUCCESS: Prefix cache is working!")
        else:
            print("WARNING: No significant speedup detected. Prefix cache may not be working.")
    
    return outputs1, outputs2


def test_output_consistency(llm, prompt, request_params):
    """Test that prefix cache doesn't affect output consistency."""
    print("\n" + "="*60)
    print("Test 3: Output Consistency Test")
    print("="*60)
    
    # Set seed for reproducibility
    request_params.seed = 42
    
    # First generation
    output1 = llm.generate([prompt], request_params, True)[0]
    
    # Second generation with same prompt (should use cache)
    output2 = llm.generate([prompt], request_params, True)[0]
    
    print(f"\nOutput 1: {output1.outputs[0].text[:200]}...")
    print(f"\nOutput 2: {output2.outputs[0].text[:200]}...")
    
    if output1.outputs[0].text == output2.outputs[0].text:
        print("\nSUCCESS: Outputs are consistent!")
    else:
        print("\nINFO: Outputs differ (expected with non-zero temperature)")
    
    return output1, output2


def test_long_prompt(llm, request_params):
    """Test with a long prompt to verify state caching at block boundaries."""
    print("\n" + "="*60)
    print("Test 4: Long Prompt Test")
    print("="*60)
    
    # Create a long prompt that spans multiple blocks
    long_prompt = """
    The history of artificial intelligence (AI) began in antiquity, with myths, 
    stories and rumors of artificial beings endowed with intelligence or 
    consciousness by master craftsmen. The seeds of modern AI were planted by 
    classical philosophers who attempted to describe the process of human thinking 
    as the mechanical manipulation of symbols. This work culminated in the invention 
    of the programmable digital computer in the 1940s, a machine based on the 
    abstract essence of mathematical reasoning. This device and the ideas behind it 
    inspired a handful of scientists to begin seriously discussing the possibility 
    of building an electronic brain.
    
    The field of AI research was founded at a workshop held on the campus of 
    Dartmouth College during the summer of 1957. Those who attended would become 
    the leaders of AI research for decades. Many of them predicted that a machine 
    as intelligent as a human being would exist in no more than a generation and 
    they were given millions of dollars to make this vision come true.
    
    Eventually, it became obvious that they had grossly underestimated the 
    difficulty of the project. In 1973, in response to the criticism from James 
    Lighthill and ongoing pressure from Congress, the U.S. and British Governments 
    stopped funding undirected research into artificial intelligence. Seven years 
    later, a visionary initiative by the Japanese Government inspired governments 
    and industry to provide AI with billions of dollars, but by the late 1980s the 
    investors became disillusioned and withdrew funding again.
    
    This cycle of boom and bust, of hype and disappointment, has been repeated 
    throughout the history of AI. However, since 2012, AI has experienced a 
    remarkable resurgence, driven by advances in machine learning, particularly 
    deep learning. Applications of AI have spread to almost every sector of 
    society, from healthcare to transportation, from finance to entertainment.
    """ * 3  # Repeat to make it longer
    
    print(f"\nPrompt length: {len(long_prompt)} characters")
    
    start_time = time.time()
    output = llm.generate([long_prompt], request_params, True)[0]
    elapsed = time.time() - start_time
    
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Generated text: {output.outputs[0].text[:200]}...")
    
    return output


def main():
    args = parse_args()
    
    print("="*60)
    print("Qwen-Next Prefix Cache Test")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Devices: {args.devices}")
    print(f"Mamba Cache Mode: {args.mamba_cache_mode}")
    print(f"Block Size: {args.block_size}")
    
    # Create LLM instance
    print("\nInitializing LLM...")
    llm = LLM(
        model=args.model,
        devices=args.devices,
        mamba_cache_mode=args.mamba_cache_mode,
        block_size=args.block_size,
        disable_prefix_cache=False,  # Enable prefix cache
    )
    
    # Create request params
    request_params = RequestParams()
    request_params.temperature = args.temperature
    request_params.top_p = args.top_p
    request_params.max_tokens = args.max_tokens
    
    # Test prompts
    basic_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    # Run tests
    test_basic_generation(llm, basic_prompts, request_params)
    
    # Test prefix cache with shared prefix
    common_prefix = "Please explain the concept of machine learning in detail. "
    unique_suffixes = [
        "What are its main applications?",
        "What are its limitations?",
        "How does it relate to AI?",
    ]
    test_prefix_cache_hit(llm, common_prefix, unique_suffixes, request_params)
    
    # Test output consistency
    test_output_consistency(llm, basic_prompts[0], request_params)
    
    # Test long prompt
    test_long_prompt(llm, request_params)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    
    llm.finish()


if __name__ == "__main__":
    main()
