# Need to have Xcode and command line tools installed!
xcode-select --install

# Install homebrew
pushd ~
which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
    brew update
fi

# Install python
which -s python
if [[ $? != 0 ]] ; then
    # Install python
    brew install python
fi

# Upgrade python packages
# FIXME: Do I need to check if user has pip installed? Is pip installed by default with any version of python?
pip install numpy --upgrade
pip install scipy --upgrade
pip install matplotlib --upgrade

# Make sure that the user's python is /usr/local/bin/python
pythonLocation=$(which python)
pythonDesiredLocation="/usr/local/bin/python"
if [ "$pythonLocation" != "$pythonDesiredLocation" ] ; then
    echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
    #echo "OpenCV needs your default Python to be located in /usr/local/bin/python"
    #echo "In other words, the command `which python` needs to output /usr/local/bin/python"
    #exit 1
fi

# Get openCV
brew tap homebrew/science
brew install opencv@2
brew install pkg-config

# Next two are bc opencv@2 is a "keg-only" package (i.e. not the most recent version)
echo 'export PATH="/sbin:$PATH" '>> ~/.bash_profile
echo 'export PATH="/usr/local/opt/opencv@2/bin:$PATH"' >> ~/.bash_profile
export PKG_CONFIG_PATH="/usr/local/opt/opencv@2/lib/pkgconfig:$PKG_CONFIG_PATH"

# Grab openCV version
opencvVersion=$(pkg-config --modversion opencv)

cat ~/.bash_profile | grep PYTHONPATH

# Symlinks for opencv to find python
ln -sf /usr/local/Cellar/opencv\@2/$opencvVersion/lib/python2.7/site-packages/cv.py cv.py
ln -sf /usr/local/Cellar/opencv\@2/$opencvVersion/lib/python2.7/site-packages/cv2.so cv2.so

source ~/.bash_profile

# Download DigiPyRo
popd
if [ ! -d "DigiPyRo" ] ; then
    git clone https://github.com/sam-may/DigiPyRo
    git checkout master
fi
