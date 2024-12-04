#!/bin/bash
for f in *.txt
do
echo "#!/bin/bash" > ${f%.*}.sh
echo -e "#SBATCH --signal=USR1@90" >> ${f%.*}.sh
echo -e "#SBATCH --job-name="${f%.*}"\t\t\t# single job name for the array" >> ${f%.*}.sh
echo -e "#SBATCH --mem=2G\t\t\t# maximum memory 100M per job" >> ${f%.*}.sh
echo -e "#SBATCH --time=01:00:00\t\t\t# maximum wall time per job in d-hh:mm or hh:mm:ss" >> ${f%.*}.sh
echo "#SBATCH --array=1-240" >> ${f%.*}.sh
echo -e "#SBATCH --account=def-ashique" >> ${f%.*}.sh

echo "cd "../../"" >> ${f%.*}.sh
echo "FILE=\"\$SCRATCH/upgd/generated_cmds/label_permuted_cifar10_stats/${f%.*}.txt\""  >> ${f%.*}.sh
echo "SCRIPT=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" \$FILE)"  >> ${f%.*}.sh
echo "module load python/3.7.9" >> ${f%.*}.sh
echo "source \$SCRATCH/upgd/.upgd/bin/activate" >> ${f%.*}.sh
echo "srun \$SCRIPT" >> ${f%.*}.sh
done