# External data (local only, not in git)

## photos/
Hotel photos for VLM coding (`biophilic vision-code`). Photos are copyrighted by the hotels / platforms, so they
stay on your machine.

```
photos/<hotel_id>/<space>/*.jpg        space = guestroom | lobby | lounge
photos/manifest.csv                    hotel_id, space, file, source_url, retrieved_on
```
`hotel_id` is the slug used everywhere (see `results/hotels.csv`). Use the same kind of official photos the 2024
judges saw (thesis §3.2: lobby, lounge, standard double room incl. window and bathroom). Sources in order of
preference: the hotel's own website, Google Places API photos, Booking.com listing photos saved manually.
Do not scrape sites whose terms forbid it.
