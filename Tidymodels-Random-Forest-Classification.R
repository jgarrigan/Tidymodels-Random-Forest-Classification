#LOAD PACKAGES
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse,    # DATA WRANGLING
               tidymodels,   # MODELLING
               vip,          # VARIABLE IMPORTANCE
               furrr,        # PARALLEL PROCESSING
               DataExplorer, # QUICK DATA EXPLORATION
               skimr,        # QUICK DATA EXPLORATION
               ranger        # RANDOM FORREST ENGINE
               )

# LOAD THE IRIS DATA
dataset <- iris

# GET SUMMARY STATISTICS FOR THE DATASET
skim(dataset)

# GET SUMMARY STATISTICS BY GROUP
aggregate(formula = . ~ Species, data = dataset, FUN = mean)

# CHECK FOR MISSINGNESS
DataExplorer::plot_missing(dataset)

# CHECK THE DISTRIBUTION OF THE DATA
DataExplorer::plot_histogram(dataset)

# CHECK CORRELATIONS BETWEEN VARIABLES
DataExplorer::plot_correlation(dataset %>% select(-Species))

# DATA PREP --------------------------------------------------------------

set.seed(111)

# CREATE A BINARY SPLIT OF DATA INTO TRAINING (75%) AND TESTING (25%)
splits <- initial_split(dataset, prop = 0.75)

# EXTRACT THE TRAINING DATAFRAME
train <- training(splits)

# EXTRACT THE TESTING DATAFRAME
test <- testing(splits)

# SPLIT THE TRAINING DATA INTO 10 GROUPS OF ROUGHLY EQUAL SIZES
cv_splits <- vfold_cv(train,
                      v = 10, 
                      strata = 'Species')

# CREATE A PRE-PROCESSING RECIPE ------------------------------------------
rec <- recipe(Species ~ .,
              data = train) %>%
  # REMOVE PREDICTOR VARIABLES EXCEEDING 0.9 PEARSON CORRELATION THRESHOLD
  step_corr(all_predictors(),
            threshold = 0.9) %>%
  # NORMALISE NUMERIC VARIABLES TO HAVE A STANDARD DEVIATION OF 1 AND MEAN OF 0
  step_normalize(all_numeric())


# DEFINE A RANDOM FOREST MODEL ----------------------------------------------------------
           # INITIALIZE A RANDOM FOREST MODEL
rf_spec <- parsnip::rand_forest() %>%
  # USE THE RANGER PACKAGE VERSION OF RANDOM FOREST
  set_engine("ranger") %>% 
  # DEFINE THE MODEL TYPE
  set_mode("classification") %>%
  set_args(mtry = tune(),  # THE NUMBER OF PREDICTORS THAT WILL BE RANDOMLY SAMPLED AT EACH SPLIT
           trees = tune(), # THE NUMBER OF TREES IN THE ENSEMBLE
           min_n = tune()) # THE MIN NUMBER OF DATA POINTS IN A NODE THAT ARE REQUIRED FOR THE NODE TO BE SPLIT FURTHER

# COMBINE PRE-PROCESSING, MODELLING AND POST-PROCESSING INTO A WORKFLOW 
rf_wf <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(rf_spec)

# SPECIFY A SEED FOR REPRODUCIBILITY
set.seed(111)

# CREATE A HYPER PARAMETER GRID TO IDENTIFY WHAT GOOD PARAMETER VALUES TO USE.
# CREATE A RANDOM GRID OF 20 COMBINATIONS
tune_res <- tune_grid(
  rf_wf,
  resamples = cv_splits,
  grid = 20
)
 
# AFTER RUNNING THE GRID ON DIFFERENT FOLDS OF THE TRAINING SET, PLOT THE 
# AUC SCORES FOR EACH PARAMETER VALUE (MIN_N,MTRY,TREES) 
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry, trees, min_n) %>%
  pivot_longer(mtry:min_n,
               values_to = "value",
               names_to = "parameter"
  )%>%
  ggplot(aes(x = value, 
             y = mean,
             color = parameter))+
  geom_point(show.legend = F)+
  facet_wrap(~ parameter,
             scales = "free_x")+
  labs(x = NULL,
       y= "AUC")

# USING A RANGE OF VALUES FROM THE GRID SEARCH BELOW THAT MAXIMIMSE THE 
# AUC SCORES LETS USE A MORE CONTROLLED GRID WITH 50 COMBINATIONS
rf_grid <- grid_regular(
  mtry(range = c(2,3)),
  trees(range = c(1000,1500)),
  min_n(range = c(3,10)),
  levels = 5 
)

# RE-COMPUTE A SET OF PERFORMANCE METRICS FOR MIN_N, MTRY, TREES USING THE OPTIMAL VALUES FROM THE PRIOR STEP
tune_res2 <- tune_grid(
  rf_wf,
  resamples = cv_splits,
  grid = rf_grid
)

# DISPLAY THE TOP SUB-MODELS AND THEIR PERFORMANCE ESTIMATES
tune_res2%>%
  show_best()

# SELECT THE BEST MODEL USING AUC METRIC
rf_best <- tune_res2 %>%
  select_best(metric = "roc_auc")

# FINALIZE A WORKFLOW WITH THE TUNED PARAMETERS
rf_wf_final <- rf_wf %>%
  finalize_workflow(rf_best)

# FIT THE MODEL AGAINST THE TEST DATA -------------------------------------
rf_fit <- rf_wf_final %>%
  last_fit(split = splits)

# OBTAIN AND FORMAT THE RESULTS OF THE TUNING FUNCTION
rf_fit %>%
  collect_metrics()

# OBTAIN AND FORMAT THE RESULTS OF THE TUNING FUNCTIONS
rf_preds <- rf_fit %>%
  collect_predictions()

rf_preds %>% 
  conf_mat(truth = Species, estimate = .pred_class)

rf_preds%>%
  roc_curve(truth = Species,
            estimate = .pred_setosa:.pred_virginica)%>%
  autoplot()
