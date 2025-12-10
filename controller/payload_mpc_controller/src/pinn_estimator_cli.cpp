#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef AUTOTRANS_WITH_TORCH
#include <torch/script.h>
#include <torch/torch.h>

namespace
{
struct CliOptions
{
  std::string model_path;
  int repeats{10};
  std::string mode{"smoke"};
  bool verbose{false};
};

void printUsage(const char *program)
{
  std::cerr << "Usage: " << program << " --model PATH [--mode smoke|benchmark] [--repeats N] [--verbose]\n";
}

CliOptions parseOptions(int argc, char **argv)
{
  CliOptions opts;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if ((arg == "--model" || arg == "-m") && i + 1 < argc)
    {
      opts.model_path = argv[++i];
    }
    else if ((arg == "--mode" || arg == "-M") && i + 1 < argc)
    {
      opts.mode = argv[++i];
    }
    else if ((arg == "--repeats" || arg == "-r") && i + 1 < argc)
    {
      opts.repeats = std::max(1, std::stoi(argv[++i]));
    }
    else if (arg == "--verbose" || arg == "-v")
    {
      opts.verbose = true;
    }
    else if (arg == "--help" || arg == "-h")
    {
      printUsage(argv[0]);
      std::exit(0);
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      std::exit(1);
    }
  }
  if (opts.model_path.empty())
  {
    std::cerr << "Missing --model argument.\n";
    printUsage(argv[0]);
    std::exit(1);
  }
  return opts;
}

torch::Tensor makeDummyInput()
{
  return torch::zeros({1, 13}, torch::TensorOptions().dtype(torch::kFloat32));
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    CliOptions opts = parseOptions(argc, argv);

    torch::jit::script::Module module = torch::jit::load(opts.model_path);
    module.to(torch::kCPU);
    module.to(torch::kFloat32);
    module.eval();
    torch::NoGradGuard guard;

    if (opts.verbose)
    {
      std::cout << "Loaded TorchScript model: " << opts.model_path << "\n";
    }

    torch::Tensor input = makeDummyInput();
    auto forward_once = [&]() {
      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(input);
      torch::Tensor output = module.forward(inputs).toTensor();
      if (opts.verbose)
      {
        std::cout << "Output tensor shape: " << output.sizes() << "\n";
      }
    };

    if (opts.mode == "smoke")
    {
      forward_once();
      std::cout << "[pinn_estimator_cli] Smoke test succeeded.\n";
    }
    else if (opts.mode == "benchmark")
    {
      std::vector<double> samples;
      samples.reserve(opts.repeats);
      for (int i = 0; i < opts.repeats; ++i)
      {
        auto start = std::chrono::steady_clock::now();
        forward_once();
        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        samples.push_back(elapsed_ms);
        if (opts.verbose)
        {
          std::cout << "Iteration " << i << ": " << elapsed_ms << " ms\n";
        }
      }
      double sum = 0.0;
      double max = 0.0;
      for (double val : samples)
      {
        sum += val;
        if (val > max)
        {
          max = val;
        }
      }
      const double avg = sum / samples.size();
      std::cout << "[pinn_estimator_cli] Ran " << samples.size() << " iterations. "
                << "Average: " << avg << " ms, Max: " << max << " ms.\n";
    }
    else
    {
      std::cerr << "Unsupported mode: " << opts.mode << "\n";
      return 2;
    }
  }
  catch (const c10::Error &e)
  {
    std::cerr << "TorchScript error: " << e.what() << "\n";
    return 3;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 4;
  }
  return 0;
}

#else

int main(int, char **)
{
  std::cerr << "[pinn_estimator_cli] LibTorch is not available in this build. "
               "Install LibTorch and configure Torch_DIR to enable this CLI.\n";
  return 1;
}

#endif
