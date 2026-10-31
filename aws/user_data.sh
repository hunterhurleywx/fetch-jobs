#!/bin/bash
# EC2 user data (Amazon Linux 2023). The launcher replaces __RUN__ with the run's S3 prefix before sending it.
# The machine: reads its job (and the R2 cache keys) from s3://hhwx-cases/__RUN__/job.env and deletes it,
# installs the pipeline, runs aws/pipeline.sh, uploads the case folder and a status file, then powers off.
# Power-off = terminate (the launcher sets InstanceInitiatedShutdownBehavior=terminate), and a hard limit
# powers it off after 150 minutes whatever happens.
RUN="__RUN__"
B=hhwx-cases
shutdown -h +150
exec > /var/log/hhwx.log 2>&1
set -x
up() { aws s3 cp /var/log/hhwx.log "s3://$B/$RUN/log.txt" --only-show-errors; }
finish() {  # finish <status>
  set +x
  up
  echo "$1" > /tmp/status && aws s3 cp /tmp/status "s3://$B/$RUN/status" --only-show-errors
  shutdown -h now
  exit 0
}
trap 'finish FAILED' ERR

aws s3 cp "s3://$B/$RUN/job.env" /root/job.env --only-show-errors
aws s3 rm "s3://$B/$RUN/job.env" --only-show-errors
set +x; set -a; . /root/job.env; set +a; rm -f /root/job.env; set -x
( while true; do sleep 30; up; done ) &

dnf install -y -q python3.12 python3.12-pip git
git clone -q https://github.com/hunterhurleywx/fetch-jobs.git /opt/pogs
cd /opt/pogs
git checkout -q "$SHA"
python3.12 -m venv /opt/venv
. /opt/venv/bin/activate
pip install -q -r requirements.txt scipy pandas shapely boto3
python -c "import eccodes; print('eccodes', eccodes.codes_get_api_version())"

export POGS_CACHE=/data/pogs_cache POGS_CASES=/opt/pogs/cases R2_BUCKET=hhwx-model-cache
mkdir -p "$POGS_CACHE"
trap - ERR
set +x
bash aws/pipeline.sh
s=$?
set -x
tar -czf /tmp/case.tar.gz -C cases "$DATE"
aws s3 cp /tmp/case.tar.gz "s3://$B/$RUN/case.tar.gz" --only-show-errors
[ $s -eq 0 ] && finish DONE || finish FAILED
