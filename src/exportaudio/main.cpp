#include <blissart/AudioObject.h>
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/RegExpValidator.h>
#include <Poco/NumberParser.h>
#include <Poco/NumberFormatter.h>
#include <iostream>
#include <vector>


using namespace blissart;
using namespace std;
using namespace Poco::Util;


class ExportAudioApp: public BasicApplication
{
public:
    ExportAudioApp()
    {
        addSubsystem(new DatabaseSubsystem);
        addSubsystem(new StorageSubsystem);
    }


protected:
    void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);

        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));

        options.addOption(
            Option("object-id", "o",
                   "Specifies a range of object IDs (id1-id2) to export",
                   true, "<id1>..<id2>", true)
            .repeatable(true)
            .validator(new RegExpValidator("\\d+(\\.\\.\\d+)?")));
    }


    void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);

        if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }

        else if (name == "object-id") {
            int min = Poco::NumberParser::parse(
                value.substr(0, value.find_first_of('.')));
            int max = Poco::NumberParser::parse(
                value.substr(value.find_last_of('.') + 1));
            for (int id = min; id <= max; ++id) {
                _objectIDs.push_back(id);
            }
        }
    }


    int main(const vector<string> &args)
    {
        if (_displayUsage || !args.empty()) {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() + " <options>\n");
            formatter.setHeader(
                "exportaudio, exports classification objects as audio files");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        DatabaseSubsystem& dbs = getSubsystem<DatabaseSubsystem>();
        for (vector<int>::const_iterator itr = _objectIDs.begin();
             itr != _objectIDs.end(); ++itr)
        {
            ClassificationObjectPtr clo = dbs.getClassificationObject(*itr);
            if (clo.isNull()) {
                cerr << "Classification object #" << *itr << " not found"
                     << endl;
            }
            else {
                cout << "Writing " << *itr << ".wav" << endl;
                audio::AudioData* pAd = AudioObject::getAudioObject(clo);
                audio::WaveEncoder::saveAsWav(*pAd, 
                    Poco::NumberFormatter::format(*itr) + ".wav");
                delete pAd;
            }
        }

        return EXIT_OK;
    }

    
private:
    bool _displayUsage;
    vector<int> _objectIDs;
};


POCO_APP_MAIN(ExportAudioApp);

