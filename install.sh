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
pip install numpy --upgrade

pythonLocation=$(which python)
pythonDesiredLocation="/usr/local/bin/python"
if [ "$pythonLocation" != "$pythonDesiredLocation" ] ; then
    echo "OpenCV needs your default Python to be located in /usr/local/bin/python"
    echo "In other words, the command `which python` needs to output /usr/local/bin/python"
    exit 1
fi

# Get openCV
brew tap homebrew/science
brew install opencv@2
brew install pkg-config

echo 'export PATH="/usr/local/opt/opencv@2/bin:$PATH"' >> ~/.bash_profile
export PKG_CONFIG_PATH="/usr/local/opt/opencv@2/lib/pkgconfig:$PKG_CONFIG_PATH"

# Grab openCV version
opencvVersion=$(pkg-config --modversion opencv)

if [ ! -f .bash_profile ] ; then
    touch .bash_profile
fi

cat ~/.bash_profile | grep PYTHONPATH
if [ -f cv.py ] ; then
    rm cv.py
fi

if [ -f cv2.so ] ; then
    rm cv2.so
fi

ln -s /usr/local/Cellar/opencv\@2/$opencvVersion/lib/python2.7/site-packages/cv.py cv.py
ln -s /usr/local/Cellar/opencv\@2/$opencvVersion/lib/python2.7/site-packages/cv2.so cv2.so

# Download DigiPyRo
popd
if [ ! -d "DigiPyRo" ] ; then
    git clone https://github.com/sam-may/DigiPyRo
    git checkout master
fi
