#mhsmm

## a shorter version of the above code
required_packages <- c("microseq", "HMM", "markovchain", "parallel", "openxlsx", "tidyverse", "doParallel", "foreach", "seqinr")
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if(length(new_packages)) install.packages(new_packages)
}

install_if_missing(required_packages)

# Load all the packages
invisible(lapply(required_packages, library, character.only = TRUE))
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
# Read the estimates object
estimates <- readRDS("permutation/estimates_parallel.rds")

simHMM <- function(hmm, length) {
  # Replace NA entries in transition or emission probabilities with zero
  hmm$transProbs[is.na(hmm$transProbs)] <- 0
  hmm$emissionProbs[is.na(hmm$emissionProbs)] <- 0
  
  # Extract relevant components for faster access
  statesVec <- hmm$States
  symbolsVec <- hmm$Symbols
  trans <- hmm$transProbs
  emiss <- hmm$emissionProbs
  
  nStates <- length(statesVec)
  nSymbols <- length(symbolsVec)
  
  # Pre-allocate output vectors
  states <- character(length)
  emission <- character(length)
  
  # Sample initial state and emission
  stateIndex <- sample.int(nStates, 1, prob = hmm$startProbs)
  states[1] <- statesVec[stateIndex]
  emissionIndex <- sample.int(nSymbols, 1, prob = emiss[stateIndex, ])
  emission[1] <- symbolsVec[emissionIndex]
  
  # Iteratively sample subsequent states and emissions
  for (i in 2:length) {
    stateIndex <- sample.int(nStates, 1, prob = trans[stateIndex, ])
    states[i] <- statesVec[stateIndex]
    emissionIndex <- sample.int(nSymbols, 1, prob = emiss[stateIndex, ])
    emission[i] <- symbolsVec[emissionIndex]
  }
  
  # Return the simulated sequence
  list(states = states, observation = emission)
}

# Function to create HMM and simulate sequences
simulate_hmm_sequences <- function(order_name, estimate) {
  # Extract emission probabilities (bimer densities) for coding and non-coding states
  bimer_density_coding <- estimate$bimer_density_coding
  bimer_density_noncoding <- estimate$bimer_density_noncoding
  
  # Ensure emission probabilities sum to 1 (normalize if necessary)
  coding_emission_probs <- bimer_density_coding / sum(bimer_density_coding, na.rm = TRUE)
  noncoding_emission_probs <- bimer_density_noncoding / sum(bimer_density_noncoding, na.rm = TRUE)
  
  # Combine the bimers into one emission probability matrix
  emission_probs <- rbind(coding_emission_probs, noncoding_emission_probs)
  
  # Get the bimer names
  bimers <- names(bimer_density_coding)
  
  # Create the HMM using the HMM package
  states <- c("coding", "non-coding")
  hmm_model <- initHMM(
    States = states,
    Symbols = bimers,
    startProbs = c(0.5, 0.5),  # Initial state probabilities (assuming equal)
    transProbs = estimate$transprob_mat,  # Transition probabilities
    emissionProbs = emission_probs  # Emission probabilities for coding and non-coding
  )
  
  # Transition matrix for markovchain simulation
  trans_matrix <- new("markovchain", 
                                   states = states, 
                                   transitionMatrix = estimate$transprob_mat)
  
  # Simulate state transitions using markovchain
  avg_len <- ceiling(estimate$avg_sequence_length / 10e5) * 10e5 / 2 * 20
  
  simulated_sequence <- simHMM(hmm_model, length = avg_len)
  
  # save the dataframe to Rds
  saveRDS(simulated_sequence, paste0("permutation/hmmdata/", order_name, ".rds"))
  return(avg_len)
}

# Loop through each order in the estimates list and simulate sequences
result <- foreach(order_name = names(estimates), .combine = 'c', .packages = c("HMM", "markovchain", "foreach")) %dopar% {
  estimate <- estimates[[order_name]]
  
  # Simulate and save the sequences for the current order
  simulate_hmm_sequences(order_name, estimate)
}

# Stop parallel cluster
stopCluster(cl)
