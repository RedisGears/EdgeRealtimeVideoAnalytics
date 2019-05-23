FROM grafana/grafana:6.1.2

ADD ./config.ini /etc/grafana/config.ini
ADD ./provisioning/dashboards /etc/grafana/provisioning/dashboards
ADD ./provisioning/datasources/prometheus-local.yaml /etc/grafana/provisioning/datasources/prometheus.yaml
ADD ./dashboards /var/lib/grafana/dashboards
