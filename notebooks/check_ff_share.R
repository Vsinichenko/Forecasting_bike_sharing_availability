library(readr)
library(data.table)
library(dplyr)
library(httr2)
library(jsonlite)

get_ff_share <- function(url) {
	req <- request(url) %>%
		req_perform()
	
	resp <- httr2::resp_body_json(req)
	

	if (length(resp$countries) > 0 && length(resp$countries[[1]]$cities) > 0) {
		l_places <- resp$countries[[1]]$cities[[1]]$places
		

		if (!is.null(l_places) && length(l_places) > 0) {
			fl_places <- lapply(l_places, function(place) {
				place[sapply(place, length) == 1]
			})
			
			dt <- rbindlist(fl_places, fill = TRUE)
			

			int_freefloats <- dt %>%
				filter(grepl("BIKE", name)) %>%
				nrow()
			int_bikes_total <- nrow(dt)
			num_share_ff <- int_freefloats / int_bikes_total
			return(list("ff_share" = num_share_ff,
									"ff" = int_freefloats,
									"total" = int_bikes_total))
		}
	}
	

	return(0)
}


req <- request("https://api.nextbike.net/maps/nextbike-live.json?list_cities=1") %>%
	req_perform()
resp <- httr2::resp_body_json(req) 
domains <- sapply(resp$countries, function(country) country$domain)
providers <- sapply(resp$countries, function(country) country$name)
countries <- sapply(resp$countries, function(country) country$country_name)


df_ff <- data.frame(
	country = character(),
	provider = character(),
	domain = character(),
	ff_share = numeric(),
	ff = integer(),
	total = integer(),
	stringsAsFactors = FALSE
)

for(i in 1:length(domains)){
	cat("i: ", i, "\n")
	base_url = "https://nextbike.net/maps/nextbike-live.json?domains="
	domain = domains[[i]]
	provider = providers[[i]]
	country = countries[[i]]
	url = paste0(base_url, domain)

	returned_list <- get_ff_share(url)
	if(class(returned_list)!="list"){
		next
	}
	num_share_ff <- returned_list$ff_share
	int_ff <- returned_list$ff
	int_total <- returned_list$total
	df_ff <- rbind(df_ff, data.frame(
		country = country,
		provider = provider,
		domain = domain,
		ff_share = num_share_ff,
		ff = int_ff,
		total = int_total,
		stringsAsFactors = FALSE
	))
}


df_ff %>%
	filter(ff_share < 0.2 & total >= 100) %>%
	arrange(ff_share)

