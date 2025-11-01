FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl cron

# Copy the cron job file
COPY forecast_cronjob /etc/cron.d/forecast-cron

# Set permissions
RUN chmod 0644 /etc/cron.d/forecast-cron && \
    crontab /etc/cron.d/forecast-cron

# Keep container running
CMD ["cron", "-f"]
