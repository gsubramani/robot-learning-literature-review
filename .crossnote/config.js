({
  // Markdown Preview Enhanced configuration for this repository.
  //
  // The textbook in documents/ writes inline math as $`\pi_\theta`$ rather than
  // plain $\pi_\theta$. The backticks are GitHub's protected inline-math form:
  // without them GitHub applies CommonMark processing to the math source before
  // rendering it, which eats backslash escapes (\{ -> {, \; -> ;, \! -> !) and
  // lets * / _ characters pair as emphasis across the $ delimiters, breaking the
  // expression entirely.
  //
  // MPE does not recognise that form by default, so it would pass the backticks
  // through to KaTeX as literal quote glyphs. Listing the pair here — before the
  // plain ["$", "$"] pair, which must stay for the fallback — makes MPE strip
  // them and render the same source correctly.
  mathInlineDelimiters: [["$`", "`$"], ["$", "$"]],

  // Display math uses ```math fences, which both GitHub and MPE render natively
  // and which are exempt from the escape-eating described above.
  mathBlockDelimiters: [["$$", "$$"]],
})
