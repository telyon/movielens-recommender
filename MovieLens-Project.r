##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download the MovieLens 10M dataset
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>%
  mutate(movieId = as.numeric(movieId),
         title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


### DATA EXPLORATION

# Inspect to see if the data is tidy
head(movielens)

# Check the variable types
str(movielens)

# Range of rating values in the dataset
range(movielens$rating)

# Histogram of rating values
ggplot(movielens, aes(rating)) +
  geom_histogram()

# Inspecting the range of years the movies were released
movielens %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  .$year %>%
  range() 

### Visualizing the general trend of ratings over the years
# Count number of ratings a movie received, group movies by year of release
ratings_per_movie <- movielens %>%
  mutate(year = str_sub(title, -5, -2)) %>%
  count(movieId, year)

# Plot number of ratings for each movie in each year
ratings_per_movie %>%
  ggplot(aes(year, n)) +
  geom_point(alpha = 0.2) + 
  ggtitle("Plot of number of ratings for individual movies released in each year") +
  xlab("Year movie was released") + 
  ylab("Number of ratings")

# Plot number of ratings for all movies released in each year
ratings_per_movie %>%
  group_by(year) %>%
  summarise(n_ratings = sum(n)) %>%     # total number of ratings in each year 
  ggplot(aes(year, n_ratings)) +
  geom_point() + 
  ggtitle("Trend of total number of ratings received by all movie released in a year") +
  ylab("Total number of ratings")

# Top 10 years with the highest total number of ratings
ratings_per_movie %>%
  group_by(year) %>%
  summarise(n_ratings = sum(n)) %>% 
  arrange(desc(n_ratings)) %>%
  head(10)


### ANALYSIS

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = F)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clear the environment
rm(dl, ratings, movies, test_index, temp, movielens, removed, ratings_per_movie)

# Split the edx dataset into training and test sets, using 80% as training data
sub_index <- createDataPartition(edx$rating, times = 1, p=0.2, list = F)
train_set <- edx[-sub_index, ]
temp <- edx[sub_index, ]

#Ensure that users and movies in the test set are also in the training set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows from test_set back into training set
removed <- anti_join(temp, test_set)
train_set <- bind_rows(train_set, removed)

# Clean up the environment
rm(sub_index, temp, removed)


###  BUILDING THE RECOMMENDER SYSTEM ALGORITHM

# Function to calculate root-mean-square error (RMSE)
RMSE <- function(predicted_value, true_value) {
  sqrt(mean((predicted_value - true_value)^2))
}

# Step 1: Randomly guessing the rating
# Create a vector of same length as test_set, with random rating values between 0.5 and 5.0
guess <- sample(seq(0.5, 5, 0.5), nrow(test_set), replace = T)
# Calculate RMSE
RMSE_guess <- RMSE(guess, test_set$rating)
# Store the result in a table
results <- data.frame(method = "Just a guess", RMSE = RMSE_guess)
results

# Step 2: Predicting the average rating across all movies
# average rating in the training set
rating_avg <- mean(train_set$rating)
# Create a vector of same length as test_set, with average rating repeated throughout
pred_avg <- rep(rating_avg, nrow(test_set))
# Calculate RMSE
RMSE_avg <- RMSE(pred_avg, test_set$rating)
# Update the result table
results <- bind_rows(results, data.frame(method = "Predicting average", RMSE = RMSE_avg))
results

# Step 3: Trying to improve prediction by incorporating (possible) movie effect
# (difference from the average rating due to the quality or nature of the movie)
# movieEffect = sum(trueRating - averageRating)/N, where N = number of ratings for the movie

# Calculate movie effect in the training set
movie_effect <- train_set %>%
  group_by(movieId) %>% 
  summarise(m_e = mean(rating - rating_avg))
# Predict rating with movie effect incorporated
pred_rating <- test_set %>% 
  left_join(movie_effect, by = "movieId") %>% 
  summarise(pred = rating_avg + m_e) %>% 
  .$pred
# Calculate RMSE
RMSE_pred_1 <- RMSE(pred_rating, test_set$rating)
# Update the results table
results <- bind_rows(results, data.frame(method = "Model incoporating movie effect", RMSE = RMSE_pred_1))
results

# Step 4: Trying to improve prediction by further incorporating (possible) user
# bias (difference from the average rating due to the individual user's personal
# preferences)
# userBias = sum(trueRating - (averageRating + movieEffect))/N, where N = number of ratings given by user

# Calculate user bias in the training set
user_bias <- train_set %>%
    left_join(movie_effect, by = "movieId") %>%
    group_by(userId) %>%
    summarise(u_b = mean(rating - rating_avg - m_e))
# Predict rating with movie effect and user bias incorporated
pred_rating <- test_set %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  summarise(pred = rating_avg + m_e + u_b) %>% 
  .$pred
# Calculate RMSE
RMSE_pred_2 <- RMSE(pred_rating, test_set$rating)
# Update the results table
results <- bind_rows(results, data.frame(method = "Model with movie effect and user bias", 
                                         RMSE = RMSE_pred_2))
results

## Exploring the data further to see if the number of ratings affects the
## difference between the true rating and the average rating.

# Plot of the influence of number of ratings on movie effect
train_set %>% group_by(movieId) %>%
  summarise(n_ratings = n()) %>%
  left_join(movie_effect, by = "movieId") %>%
  ggplot(aes(n_ratings, abs(m_e))) + 
  geom_line() +
  ggtitle("Plot of movie effect as affected by number of ratings") + 
  ylab("Movie effect") +
  xlab("Number of ratings")

# Plot of the influence of number of ratings on user bias
train_set %>% group_by(userId) %>%
  summarise(n_ratings = n()) %>%
  left_join(user_bias, by = "userId") %>%
  ggplot(aes(n_ratings, abs(u_b))) + 
  geom_line() +
  ggtitle("Plot of user bias as affected by number of ratings") +
  ylab("User bias") +
  xlab("Number of ratings")
  
# Step 5: Regularizing the effect of unpopular movies

# range of lambdas to find which one minimizes the RMSE
lambdas <- seq(1,10)

# Create function to calculate regularized RMSE for a given value of lambda using
# movieEffectReg = sum(trueRating - averageRating)/(N + lambda)
# userBiasReg = sum(trueRating - (averageRating + movieEffect))/(N + lambda)
RMSE_reg <- function(l) {
  # Regularize the movie effect
  movie_effect_reg <- train_set %>%
    group_by(movieId) %>%
    summarise(m_e_reg = sum(rating - rating_avg)/(n() + l))
  # Regularize the user bias
  user_bias_reg <- train_set %>%  
    left_join(movie_effect_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarise(u_b_reg = sum(rating - rating_avg - m_e_reg)/(n() + l))
  # Predict rating with regularized movie effect and user bias incorporated
  pred_rating <- test_set %>%
    left_join(movie_effect_reg, by = "movieId") %>%
    left_join(user_bias_reg, by = "userId") %>%
    summarise(pred = rating_avg + m_e_reg + u_b_reg) %>%
    .$pred
  # Calculate RMSE
  RMSE(pred_rating, test_set$rating)
}

# Apply the function to the selected range of lambda values
reg <- data.frame(lambda = lambdas, RMSE = sapply(lambdas, RMSE_reg))

# Plot RMSEs of the regularized model against lambda 
reg %>% ggplot(aes(lambda, RMSE)) +
  geom_line()

# Select the value of lambda that best minimizes the RMSE
lambda <- lambdas[which.min(reg$RMSE)]

# Update the results table with the best (lowest) RMSE
results <- bind_rows(results, data.frame(method = "Improved model with regularization", 
                                         RMSE = min(reg$RMSE)))
results

# Applying this last algorithm in Step 5 to predict the values in the validation data set

# Using the best value of lambda; with edx as the training data for the algorithm
# Recalculate the regularized movie effect
movie_effect_reg <- edx %>%
  group_by(movieId) %>%
  summarise(m_e_reg = sum(rating - rating_avg)/(n() + lambda))
# Recalculate the regularized user bias
user_bias_reg <- edx %>%  
  left_join(movie_effect_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarise(u_b_reg = sum(rating - rating_avg - m_e_reg)/(n() + lambda))
# Predict rating in the validation data
pred_validation <- validation %>%
  left_join(movie_effect_reg, by = "movieId") %>%
  left_join(user_bias_reg, by = "userId") %>%
  summarise(pred = rating_avg + m_e_reg + u_b_reg) %>%
  .$pred
# Calculate RMSE
RMSE_validation <- RMSE(pred_validation, validation$rating)
RMSE_validation

# Update the results with the RMSE obtained form the validation data
results <- bind_rows(results, data.frame(method = "Improved model applied to the validation set", 
                                         RMSE = RMSE_validation))
# Calculate the percentage improvement for each step in the model development
results <- mutate(results, "% improvement" = (RMSE[1] - RMSE)*100 / RMSE[1])
results <- bind_cols(data.frame(step = c(1:6)), results)
results
