from models.audio import AudioEncoder
from models.lang import LM
import librosa


if __name__ == '__main__':

    # TODO implement data fetching method

    # GPT example
    lm = LM()

    lyrics = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
    )

    text_embedding = lm.encode(lyrics)
    lm.decode(text_embedding)

    # Wav2Vec2 example
    subclip_path = "/Users/omnipede/data/2Pac-Changes(OfficialMusicVideo)ft.Talent/clips/2Pac-Changes(OfficialMusicVideo)ft.Talent-99766-2235.mp4"
    subclip, _ = librosa.load(subclip_path)

    audio_encoder = AudioEncoder()
    embed = audio_encoder.encode(subclip)

    print(embed)
    print(embed.size())

    pass
