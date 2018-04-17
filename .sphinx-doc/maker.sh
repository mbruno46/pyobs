
cp conf.py home
sed -e 's/index/tutorials/' conf.py > tutorials/conf.py
sed -e 's/index/pyobs/' conf.py > pyobs/conf.py

cd home
make html
cd ../tutorials
make html
cd ../pyobs
make html
