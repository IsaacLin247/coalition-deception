-- Group the manuscript's manually numbered captions with their images/tables.
-- The caption text is preserved verbatim as Pandoc inlines; LaTeX suppresses its
-- automatic label so the existing Figure 1--19 and Table 1--17/B1 labels appear once.
local figures, tables = 0, 0

local function caption_number(block, kind)
  if not block or block.t ~= 'Para' then return nil end
  return pandoc.utils.stringify(block):match('^' .. kind .. ' ([%w]+)%. ')
end

local function latex_table_float(block, caption, number)
  -- Pandoc's LaTeX writer uses longtable even for short tables. A genuine table
  -- float avoids longtable's conflicting output routine on pages with figures.
  -- Retain its generated cell contents/column specs; remove only continuation
  -- headers/footers, and place the unchanged caption inside the float.
  block.caption.long = {}
  block.identifier = ''
  local latex = pandoc.write(pandoc.Pandoc({block}), 'latex')
  local first, body = latex:match('^(.-)\\endhead.-\\endlastfoot(.*)$')
  assert(first and body, 'Unexpected Pandoc longtable structure for Table ' .. number)
  first = first:gsub('\\begin{longtable}%b[]', '\\begin{tabular}', 1)
  body = body:gsub('\\end{longtable}', '\\bottomrule\\noalign{}\n\\end{tabular}', 1)
  local caption_tex = pandoc.write(pandoc.Pandoc({caption}), 'latex'):gsub('\n+$', '')
  local placement = number == '17' and 'p' or 'htbp'
  return pandoc.RawBlock('latex', '\\begin{table}[' .. placement .. ']\n\\centering\n' ..
    '\\caption{' .. caption_tex .. '}\\label{tab:' .. number .. '}\n' .. first .. body ..
    '\n\\end{table}')
end

local function group_captions(blocks)
  local out, i = pandoc.List(), 1
  while i <= #blocks do
    local block, following = blocks[i], blocks[i + 1]
    local figure_number = caption_number(following, 'Figure')
    local table_number = caption_number(following, 'Table')
    if block.t == 'Para' and #block.content == 1 and
        block.content[1].t == 'Image' and figure_number then
      local img = block.content[1]
      -- Bound the image while leaving room for its 12-point caption on one page.
      img.attributes.width = img.attributes.width or '100%'
      img.attributes.height = img.attributes.height or '65%'
      out:insert(pandoc.Figure({block}, {following},
        pandoc.Attr('fig:' .. figure_number)))
      figures = figures + 1
      i = i + 2
    elseif block.t == 'Table' and table_number then
      block.caption.long = {following}
      block.identifier = 'tab:' .. table_number
      local specs = block.colspecs
      if table_number == '3' or table_number == '4' then
        -- Give the unbreakable 'Median' heading its required width at 12 points.
        specs[1][2] = specs[1][2] - 0.005
        specs[3][2] = specs[3][2] + 0.005
        block.colspecs = specs
      elseif table_number == '17' then
        local widths = {0.20, 0.085, 0.055, 0.10, 0.10, 0.10, 0.10, 0.105, 0.155}
        for column, width in ipairs(widths) do specs[column][2] = width end
        block.colspecs = specs
      end
      if FORMAT:match('latex') then
        -- Preserve manual figure/table ordering across the independent float queues.
        out:insert(pandoc.RawBlock('latex', '\\FloatBarrier'))
      end
      if table_number == '17' and FORMAT:match('latex') then
        -- Nine columns and a long caption fit at the existing 12-point size on
        -- a dedicated landscape page; do not leave a one-row continuation.
        out:insert(pandoc.RawBlock('latex', '\\begin{landscape}'))
        out:insert(latex_table_float(block, following, table_number))
        out:insert(pandoc.RawBlock('latex', '\\end{landscape}'))
      elseif (table_number == '6' or table_number == 'B1') and FORMAT:match('latex') then
        -- Complete mechanism cells and the configuration appendix genuinely
        -- span pages. Earlier floats have been flushed; retain continuation headers.
        out:insert(block)
      elseif FORMAT:match('latex') then
        out:insert(latex_table_float(block, following, table_number))
      else
        out:insert(block)
      end
      tables = tables + 1
      i = i + 2
    else
      if block.t == 'Header' and block.level <= 2 and FORMAT:match('latex') then
        out:insert(pandoc.RawBlock('latex', '\\FloatBarrier'))
        if pandoc.utils.stringify(block):match('^Appendix B%.') then
          -- Start the genuinely multipage configuration table with enough room
          -- for its appendix heading, caption, and several rows together.
          out:insert(pandoc.RawBlock('latex', '\\clearpage'))
        end
      end
      out:insert(block)
      i = i + 1
    end
  end
  return out
end

return {
  { Blocks = group_captions },
  { Pandoc = function(doc)
      assert(figures == 19, 'Expected 19 manually captioned figures; found ' .. figures)
      assert(tables == 18, 'Expected 18 manually captioned tables; found ' .. tables)
      return doc
    end },
}
