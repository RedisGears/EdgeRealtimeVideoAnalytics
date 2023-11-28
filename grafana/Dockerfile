FROM grafana/grafana:6.7.5-ubuntu

ADD ./config.ini /etc/grafana/config.ini
ADD ./provisioning/dashboards /etc/grafana/provisioning/dashboards
ADD ./provisioning/datasources/prometheus.yaml /etc/grafana/provisioning/datasources/prometheus.yaml
ADD ./dashboards /var/lib/grafana/dashboards
