-- Keep the manually numbered captions attached to the five figures/three tables.
-- This independent filter leaves the original manuscript's layout unchanged.
local figures, tables = 0, 0

local function caption_number(block, kind)
  if not block or block.t ~= 'Para' then return nil end
  return pandoc.utils.stringify(block):match('^' .. kind .. ' (%d+)%. ')
end

local function table_float(block, caption, number)
  block.caption.long = {}
  block.identifier = ''
  local latex = pandoc.write(pandoc.Pandoc({block}), 'latex')
  local first, body = latex:match('^(.-)\\endhead.-\\endlastfoot(.*)$')
  assert(first and body, 'Unexpected Pandoc table structure: ' .. number)
  first = first:gsub('\\begin{longtable}%b[]', '\\begin{tabular}', 1)
  body = body:gsub('\\end{longtable}', '\\bottomrule\\noalign{}\n\\end{tabular}', 1)
  local text = pandoc.write(pandoc.Pandoc({caption}), 'latex'):gsub('\n+$', '')
  return pandoc.RawBlock('latex', '\\begin{table}[htbp]\n\\centering\n' ..
    '\\caption{' .. text .. '}\\label{tab:' .. number .. '}\n' .. first .. body ..
    '\n\\end{table}')
end

local function group_captions(blocks)
  local out, i = pandoc.List(), 1
  while i <= #blocks do
    local block, following = blocks[i], blocks[i + 1]
    local fn = caption_number(following, 'Figure')
    local tn = caption_number(following, 'Table')
    if block.t == 'Para' and #block.content == 1 and block.content[1].t == 'Image' and fn then
      block.content[1].attributes.height = fn == '5' and '75%' or '60%'
      out:insert(pandoc.Figure({block}, {following}, pandoc.Attr('fig:' .. fn)))
      figures = figures + 1
      i = i + 2
    elseif block.t == 'Table' and tn then
      block.caption.long = {following}
      block.identifier = 'tab:' .. tn
      local widths
      if tn == '1' then widths = {0.23, 0.77}
      elseif tn == '2' then widths = {0.17, 0.11, 0.15, 0.18, 0.18, 0.21}
      elseif tn == '3' then widths = {0.18, 0.17, 0.20, 0.20, 0.14, 0.11} end
      for k, width in ipairs(widths) do block.colspecs[k][2] = width end
      if FORMAT:match('latex') then
        out:insert(pandoc.RawBlock('latex', '\\FloatBarrier'))
        out:insert(table_float(block, following, tn))
      else out:insert(block) end
      tables = tables + 1
      i = i + 2
    else
      if block.t == 'Header' and block.level <= 2 and FORMAT:match('latex') then
        out:insert(pandoc.RawBlock('latex', '\\FloatBarrier'))
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
      assert(figures == 5, 'Expected 5 figures; found ' .. figures)
      assert(tables == 3, 'Expected 3 tables; found ' .. tables)
      return doc
    end },
}
